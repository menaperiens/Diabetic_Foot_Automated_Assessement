"""
Horizontal toe alignment.

Unlike the bridge and vertical-toe scripts, this one does not run an
iterative capture-measure-move loop (see alignment_common.align_axis).
It captures a single image, estimates where each of the five toes is
(with interactive confirmation via pygame), and moves the horizontal
sensor once to the big toe's column. That one-shot shape is intrinsic
to the algorithm, not something to force into align_axis -- so this
script shares only what's genuinely common with the other scripts
(logging setup, debug folder convention, background segmentation, and
consistent error handling / confirm-before-move), not the control loop.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pygame
from PIL import Image, ImageDraw
from rembg import remove

from alignment_common import AlignmentError, CaptureFailedError, LaserNotDetectedError, get_logger, make_debug_folder, segment_foot
from utils import HorizontalDirection, Motor, capture_save, laser_dot_detec_hrz, retry

logger = get_logger(__name__, level=logging.INFO)

TOE_MASK_ALPHA_THRESHOLD = 200
# When trying background removal with alpha matting, it was verified that
# a good threshold for a pixel to be considered as belonging to the foot was
# 200. (Initially the threshold picked was 254. However, the foot edges are
# dimmer (weaker) than 254, and the black foot support pixels have values
# >254, which broke pinky-toe detection since the support sat above the foot
# edge on its left, so the loop never crossed the threshold there.)

LASER_TO_SENSOR_OFFSET = 279 - 92 + 50
# In one picture we found that the row of the center of the sensor was 92
# whereas the row of the centroid of the laser was 279. 50 was added later,
# upon calibration refinement. In some pictures the sensor was not visible
# (this is when the sensor was at the posterior limit).

STEP_SIZE_PX = 1.6  # tuned empirically for this setup, like the other calibration constants in this codebase.


# --------------------------------------------------------------------------
# Toe-line drawing helpers (pygame/PIL)
# --------------------------------------------------------------------------

def build_pil_image(img, middle_line_l, middle_line_r, side, left_margin, right_margin, col_of_first_true):
    """Build a PIL image with the drawn lines, using the current middle-line values."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    if side == "right":
        draw.line([(col_of_first_true, middle_line_l), (right_margin, middle_line_r)], fill="red", width=2)
        draw.line([(left_margin, middle_line_l), (col_of_first_true, middle_line_l)], fill="red", width=2)
    elif side == "left":
        draw.line([(left_margin, middle_line_l), (col_of_first_true, middle_line_r)], fill="red", width=2)
        draw.line([(col_of_first_true, middle_line_r), (right_margin, middle_line_r)], fill="red", width=2)
    else:
        raise ValueError("side must be 'left' or 'right'")
    return img_copy


def build_surface(img, middle_line_l, middle_line_r, side, left_margin, right_margin, col_of_first_true):
    """Build a pygame surface from the PIL image with the drawn lines."""
    img_copy = build_pil_image(img, middle_line_l, middle_line_r, side, left_margin, right_margin, col_of_first_true)
    mode = img_copy.mode
    size = img_copy.size
    data = img_copy.tobytes()
    return pygame.image.fromstring(data, size, mode)


def draw_dot(draw, x, y, r=5, fill="yellow", outline="black"):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline)


def build_pil_image_ind_toe(img, row, col):
    """Build a PIL image with an individual toe marked."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw_dot(draw, col, row)
    return img_copy


def build_surface_ind_toe(img, row, col):
    """Build a pygame surface from the PIL image with an individual toe marked."""
    img_copy = build_pil_image_ind_toe(img, row, col)
    mode = img_copy.mode
    size = img_copy.size
    data = img_copy.tobytes()
    return pygame.image.fromstring(data, size, mode)


def red_line_y(x, side, left_margin, right_margin, col_of_first_true, middle_line_l, middle_line_r):
    """Given an x position, what should the y of the red line be?"""
    if side == "right":
        if x <= col_of_first_true:
            # If x is left of the first (left-most) foot pixel, the toe line
            # is treated as flat (this accounts for feet whose toe line isn't
            # strictly ascending -- it can flatten or even decrease at the end).
            return middle_line_l
        else:
            # Right of the first foot pixel, the toe line is assumed strictly
            # ascending, so interpolate linearly between the two endpoints.
            t = (x - col_of_first_true) / (right_margin - col_of_first_true)
            return middle_line_l + t * (middle_line_r - middle_line_l)

    if side == "left":
        if x <= col_of_first_true:
            t = (x - left_margin) / (col_of_first_true - left_margin)
            return middle_line_l + t * (middle_line_r - middle_line_l)
        else:
            return middle_line_r
    else:
        raise ValueError("side must be 'left' or 'right'")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    debug_folder = make_debug_folder("horizontal_toe_alignment")
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    motor = Motor()

    try:
        # capture image from camera and save it to file, returning the file path
        file = retry(capture_save, "top", debug_folder)
        if file is None:
            raise CaptureFailedError("Failed to capture frame for horizontal toe alignment")

        logger.info("analysing file: %s", file)
        input_img = file.read_bytes()

        # Remove background and get an image with an alpha channel (foot
        # opaque, background transparent). alpha_matting=True is used here
        # (rather than a pre-built session, as the bridge/vertical-toe
        # scripts use) because it helps when the sensors overlap the foot --
        # without it, the sensor is sometimes treated as foreground.
        img, alpha = segment_foot(input_img, alpha_matting=True)

        mask = alpha > TOE_MASK_ALPHA_THRESHOLD
        rows = np.any(mask, axis=1)  # per-row: True if that row contains any foot pixel

        foot_finish = np.where(rows)[0][-1]  # last row with foot, which corresponds to the ankle area.
        logger.debug("Foot finishes at row: %d", foot_finish)

        columns = np.any(mask, axis=0)  # columns that contain foot

        left_margin = np.argmax(columns)  # first column with foot -> left edge of the foot;
        # reference point for toe positions when side == "right".
        right_margin = np.where(columns)[0][-1]  # last column with foot -> right edge of the foot;
        # reference point for toe positions when side == "left".

        width = right_margin - left_margin

        first_row = np.argmax(rows)  # first row with foot
        logger.debug("First row with foot: %d", first_row)

        cols_in_first_row = np.where(mask[first_row])[0]  # columns where the first row with foot is True

        col_of_first_true = cols_in_first_row[0]  # first (left-most) column in that row: starting point
        # to determine if the foot is on the left or right side of the body.

        side = "left" if col_of_first_true > left_margin + width / 2 else "right"
        # If the first (left-most) pixel in the top-most row is in the right half of the
        # foot, the foot is on the left side of the body, otherwise on the right side.
        # The big toe (top-most pixel) is on the left side of the foot if the foot is on
        # the right side of the body, and on the right side of the foot if the foot is
        # on the left side of the body.

        if side == "left":
            col_of_first_true = cols_in_first_row[-1]  # if left foot, take the right-most True pixel
        logger.debug("Column of first true pixel in the first row with foot pixels: %d", col_of_first_true)

        threshold_constant = 0.4
        # Fraction of foot length (measured from the ankle) that defines the row where
        # pinky-toe search begins. Naively taking the left-most/right-most pixel often
        # doesn't give the actual toe -- in many cases it was the start of the leg
        # instead, because the foot was tilted. Restricting the search to the upper
        # part of the foot avoids this. May need tuning for a different population.

        threshold = int(threshold_constant * (foot_finish - first_row))
        logger.debug("Threshold for pinky toe start: %d", threshold)

        offset_tolerance = 0.1
        # Starting distance (as a proportion of foot width) from the foot's edge at
        # which to begin searching for the toe start. Grows each loop iteration to
        # search progressively further inward.

        toe_start_right = 719  # sentinel start value = image height (720px) - 1
        toe_start_left = 719   # search loops below move this upward (lower row numbers)
        # as they search inward for the toe.

        # Goal: find the row of the big toe and pinky toe. The loops below search for
        # the top, right-most (if foot on the right) or left-most (if foot on the left)
        # foot pixel (pinky toe location), which must be above the row threshold
        # computed above. Starting from the outer edge, we move inward column by
        # column until the threshold is crossed or the estimate stops improving.
        if side == "right":
            toe_start_left = first_row  # the left-most toe is easy: it's the highest (lowest-row) point in the foot

            while toe_start_right > threshold:
                offset = int(offset_tolerance * width)
                logger.debug("offset: %d", offset)

                col_right = mask[:, right_margin - offset]
                logger.debug("col_right: %d", right_margin - offset)

                if np.any(col_right):
                    toe_start_right_new = np.argmax(col_right)
                    offset_tolerance += 0.01

                    if toe_start_right_new <= toe_start_right:
                        toe_start_right = toe_start_right_new
                        logger.debug("Current estimate for toe (right) start row: %d", toe_start_right)
                    else:
                        logger.debug("breaking loop")
                        logger.info("Toe (right) starts at row: %d", toe_start_right)
                        break
                else:
                    offset_tolerance += 0.02
            logger.info("Toe (left) starts at row: %d", toe_start_left)

        elif side == "left":
            toe_start_right = first_row
            while toe_start_left > threshold:
                offset = int(offset_tolerance * width)
                col_left = mask[:, left_margin + offset]
                if np.any(col_left):
                    toe_start_left_new = np.argmax(col_left)
                    offset_tolerance += 0.01
                    logger.debug("Current estimate for toe (left) start row: %d", toe_start_left_new)
                    if toe_start_left_new <= toe_start_left:
                        toe_start_left = toe_start_left_new
                    else:
                        logger.debug("breaking loop")
                        logger.info("Toe (left) starts at row: %d", toe_start_left)
                        break
                else:
                    offset_tolerance += 0.02
            logger.info("Toe (right) starts at row: %d", toe_start_right)

        else:
            raise ValueError("side must be 'left' or 'right'")

        # Foot height on each side: from the corresponding toe start row to the
        # foot's finish row. Used as a reference length to locate where the toe
        # ends, since toes occupy a known fraction of this height near the top.
        foot_height_r = foot_finish - toe_start_right
        foot_height_l = foot_finish - toe_start_left

        # Toe height estimated as a proportion of foot height, based on the average
        # foot shape/size in the population this was tuned on.
        if side == "right":
            toe_height_r = foot_height_r * 0.187
            toe_finish_right = int(toe_start_right + toe_height_r)
            logger.info("Toe (right) finishes at row: %d", toe_finish_right)

            toe_height_l = foot_height_l * 0.270
            toe_finish_left = int(toe_start_left + toe_height_l)
            logger.info("Toe (left) finishes at row: %d", toe_finish_left)

        elif side == "left":
            toe_height_r = foot_height_r * 0.318
            toe_finish_right = int(toe_start_right + toe_height_r)
            logger.info("Toe (right) finishes at row: %d", toe_finish_right)

            toe_height_l = foot_height_l * 0.224
            toe_finish_left = int(toe_start_left + toe_height_l)
            logger.info("Toe (left) finishes at row: %d", toe_finish_left)

        else:
            logger.error("Invalid side value: %s", side)
            raise ValueError("side must be 'left' or 'right'")

        draw = ImageDraw.Draw(img)

        # Target folder for marked images, named with the current date/time to
        # avoid overwriting previous markings and to keep track of when they were made.
        target_folder = debug_folder / f"marked  {start_time}"
        target_folder.mkdir(exist_ok=True)

        if side == "right":
            draw.line([(col_of_first_true, toe_start_left), (right_margin, toe_start_right)], fill="blue", width=2)
            draw.line([(left_margin, toe_start_left), (col_of_first_true, toe_start_left)], fill="blue", width=2)
            draw.line([(col_of_first_true, toe_finish_left), (right_margin, toe_finish_right)], fill="blue", width=2)
            draw.line([(left_margin, toe_finish_left), (col_of_first_true, toe_finish_left)], fill="blue", width=2)

            middle_line_r = toe_start_right + toe_height_r / 2
            middle_line_l = toe_start_left + toe_height_l / 2

        elif side == "left":
            draw.line([(left_margin, toe_start_left), (col_of_first_true, toe_start_right)], fill="blue", width=2)
            draw.line([(col_of_first_true, toe_start_right), (right_margin, toe_start_right)], fill="blue", width=2)
            draw.line([(left_margin, toe_finish_left), (col_of_first_true, toe_finish_right)], fill="blue", width=2)
            draw.line([(col_of_first_true, toe_finish_right), (right_margin, toe_finish_right)], fill="blue", width=2)

            middle_line_r = toe_start_right + toe_height_r / 2
            middle_line_l = toe_start_left + toe_height_l / 2

        else:
            raise ValueError("side must be 'left' or 'right'")

        input(
            "Se necessario, ajuste a posicao da linha vermelha, por favor. "
            "Esta deve passar pelo centro de todos os dedos. Presse Enter para visualizar."
        )

        logger.info("Start of interactive mode.")
        pygame.init()
        screen = pygame.display.set_mode((800, 600))

        start = time.time()
        TIMEOUT = 60  # seconds
        running = True

        while running:
            if time.time() - start > TIMEOUT:
                logger.error("Foot line adjustment timed out")
                print("Tempo para ajuste da linha excedido. Continuando.")
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                if keys[pygame.K_UP]:
                    middle_line_l -= 3 * toe_height_l // 100
                if keys[pygame.K_DOWN]:
                    middle_line_l += 3 * toe_height_l // 100
            if keys[pygame.K_RIGHT]:
                if keys[pygame.K_UP]:
                    middle_line_r -= 3 * toe_height_r // 100
                if keys[pygame.K_DOWN]:
                    middle_line_r += 3 * toe_height_r // 100

            surface = build_surface(img, middle_line_l, middle_line_r, side, left_margin, right_margin, col_of_first_true)

            img_w, img_h = surface.get_size()
            screen_w, screen_h = screen.get_size()
            scale = min(screen_w / img_w, screen_h / img_h)
            new_size = (int(img_w * scale), int(img_h * scale))
            surface = pygame.transform.scale(surface, new_size)

            x = (screen_w - new_size[0]) // 2
            y = (screen_h - new_size[1]) // 2

            screen.fill((255, 255, 255))
            screen.blit(surface, (x, y))
            pygame.display.flip()

        final_middle_line_l = middle_line_l
        final_middle_line_r = middle_line_r

        pygame.quit()
        target_file = target_folder / f"marked_{file.name}"
        img.save(target_file)
        logger.info("Image saved to %s", target_file)

        # Estimate initial column positions for each toe, based on foot width and
        # side of body, as a starting point for interactive marking. Row position
        # for all toes starts on the middle (red) line.
        if side == "right":
            big_toe_col = left_margin + 1 / 8 * width
            index_toe_col = big_toe_col + (1 / 8 + (3 / 4 * 1 / 4) / 2) * width
            middle_toe_col = index_toe_col + 3 / 16 * width
            ring_toe_col = middle_toe_col + 3 / 16 * width
            pinky_toe_col = ring_toe_col + 3 / 16 * width
        else:
            big_toe_col = right_margin - 1 / 8 * width
            index_toe_col = big_toe_col - (1 / 8 + (3 / 4 * 1 / 4) / 2) * width
            middle_toe_col = index_toe_col - 3 / 16 * width
            ring_toe_col = middle_toe_col - 3 / 16 * width
            pinky_toe_col = ring_toe_col - 3 / 16 * width

        toe_dict = {
            "big": (big_toe_col, None),
            "index": (index_toe_col, None),
            "middle": (middle_toe_col, None),
            "ring": (ring_toe_col, None),
            "pinky": (pinky_toe_col, None),
        }

        logger.info("dictionary of toe positions before refinement: %s", toe_dict)
        input(
            "Confirme a posicao de cada dedo, por favor. \n"
            "Utilize a as setas direita e esquerda para mover o ponto amarelo e presse Enter quando o ponto "
            "estiver na posicao certa. Isto deve ser feito para cada um dos dedos. Presse Enter para comecar."
        )

        pygame.init()
        screen = pygame.display.set_mode((800, 600))

        for toe, (toe_col, toe_row) in toe_dict.items():
            toe_row = red_line_y(toe_col, side, left_margin, right_margin, col_of_first_true, final_middle_line_l, final_middle_line_r)

            start = time.time()
            TIMEOUT = 60
            running = True

            while running:
                if time.time() - start > TIMEOUT:
                    logger.error("Foot line adjustment timed out")
                    print("Tempo para ajuste da linha excedido. Continuando.")
                    running = False

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            img = build_pil_image_ind_toe(img, toe_row, toe_col)
                            running = False
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break

                keys = pygame.key.get_pressed()

                if keys[pygame.K_LEFT]:
                    toe_col -= 1 * width // 100
                    toe_row = red_line_y(toe_col, side, left_margin, right_margin, col_of_first_true, final_middle_line_l, final_middle_line_r)
                if keys[pygame.K_RIGHT]:
                    toe_col += 1 * width // 100
                    toe_row = red_line_y(toe_col, side, left_margin, right_margin, col_of_first_true, final_middle_line_l, final_middle_line_r)

                surface = build_surface_ind_toe(img, toe_row, toe_col)

                img_w, img_h = surface.get_size()
                screen_w, screen_h = screen.get_size()
                scale = min(screen_w / img_w, screen_h / img_h)
                new_size = (int(img_w * scale), int(img_h * scale))
                surface = pygame.transform.scale(surface, new_size)

                x = (screen_w - new_size[0]) // 2
                y = (screen_h - new_size[1]) // 2

                screen.fill((255, 255, 255))
                screen.blit(surface, (x, y))
                pygame.display.flip()

                toe_dict[toe] = (toe_col, toe_row)

            img.save(target_folder / f"marked_toes_{file.name}")

        pygame.quit()

        logger.info("dictionary of toe positions after refinement: %s", toe_dict)

        laser_c_row = retry(laser_dot_detec_hrz, file)
        if laser_c_row is None:
            # The original script used laser_c_row unconditionally here, which
            # would raise a confusing TypeError deep in string formatting if the
            # laser wasn't found, rather than failing clearly. Now it fails the
            # same way the other alignment scripts do.
            logger.error("Laser dot not detected in image.")
            raise LaserNotDetectedError("Laser dot not detected during horizontal toe alignment")

        logger.debug("position of laser: %f", laser_c_row)

        sensor_ini_row = laser_c_row - LASER_TO_SENSOR_OFFSET
        logger.debug("Initial position (row) of the sensor: %f", sensor_ini_row)

        # Only the big toe's row is used for the horizontal motor move below.
        # index/middle/ring/pinky are marked interactively for visual verification
        # (and potential future use) but aren't currently consumed elsewhere.
        _, big_toe_row = toe_dict["big"]
        distance_motor = big_toe_row - sensor_ini_row
        logger.debug("Distance to be traversed (px): %f", distance_motor)
        # if negative -> sensor is too far into the foot. needs to go "FRONT" (away from ankle)
        # if positive -> sensor is too far from the foot. needs to go "BACK" (towards ankle)

        n_steps = int(abs(distance_motor / STEP_SIZE_PX))
        direction = HorizontalDirection.BACK if distance_motor > 0 else HorizontalDirection.FRONT

        # Confirmation prompt added for consistency with the bridge and
        # vertical-toe scripts, which all pause before moving the motor.
        input("Sensor prestes a mexer. Pressione Enter para continuar.")
        logger.info("Moving horizontal motor. Direction=%s Steps=%d", direction.name, n_steps)
        try:
            motor.move(direction, n_steps)
            logger.info("Horizontal motor movement completed")
        except Exception:
            logger.exception("Horizontal toe positioning failed")

    except AlignmentError as exc:
        logger.exception("Horizontal toe alignment aborted")
        sys.exit(exc.exit_code)
    finally:
        motor.close()


if __name__ == "__main__":
    main()
