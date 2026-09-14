"""
Shared primitives for foot-relative sensor alignment.

Both `foot_bridge_align.py` (bridge / peito do pe, vertical + horizontal)
and `toe_alignment.py` (toe row, vertical) follow the same pattern:

    capture -> remove background -> find a target row in the foot mask
    -> detect the laser dot (current sensor position) -> compute the
    pixel distance -> move the motor -> repeat until close enough,
    stalled, or out of iterations.

This module holds everything that pattern has in common, parameterized
per script/axis via `AxisConfig`, so a fix made here (e.g. the stall
-detection logic) applies to every script that uses it instead of having
to be found and re-fixed independently in each copy.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw
from rembg import remove

from utils import Motor, capture_save, retry


# Exit codes shared across every script that uses this module (and by
# move_sensor_limit_horizontal.py, which doesn't use align_axis but should
# still fail consistently): 10 for laser detection failure, 11 for image
# capture failure, 12 for a horizontal movement failure (e.g. homing timeout).
EXIT_LASER_NOT_DETECTED = 10
EXIT_CAPTURE_FAILED = 11
EXIT_MOVEMENT_FAILED = 12


class AlignmentError(Exception):
    """Base class for alignment failures. Subclasses set `exit_code` so a
    caller can do `sys.exit(exc.exit_code)` without needing to know which
    specific failure occurred."""
    exit_code = 1


class CaptureFailedError(AlignmentError):
    """Raised when a frame could not be captured after retries."""
    exit_code = EXIT_CAPTURE_FAILED


class LaserNotDetectedError(AlignmentError):
    """Raised when the laser dot could not be detected after retries."""
    exit_code = EXIT_LASER_NOT_DETECTED


# --------------------------------------------------------------------------
# Logging
#
# Each script gets its own logger with its own handler/level instead of all
# calling logging.basicConfig() themselves. basicConfig() only takes effect
# on the *first* call in a process, so two scripts each calling it with a
# different level (as the originals did: INFO vs DEBUG) would silently let
# the second one's level be ignored if they ever ran in the same process.
# --------------------------------------------------------------------------

def get_logger(module_name: str, level: int = logging.INFO, log_file: str = "main.log") -> logging.Logger:
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Debug output
# --------------------------------------------------------------------------

def make_debug_folder(subfolder_name: str) -> Path:
    """Create (and return) a fresh timestamped debug folder for this run."""
    parent = Path.home() / "Documents" / "DFAA_debug" / subfolder_name
    parent.mkdir(parents=True, exist_ok=True)
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    folder = parent / start_time
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_debug_line(
    img: Image.Image,
    x0: int,
    x1: int,
    y: int,
    folder: Path,
    name: str,
    overwrite: bool = False,
) -> None:
    """
    Draw and save a horizontal line marking a detected target row, for
    debugging. By default each save gets its own timestamp (so older
    debug images are kept); pass overwrite=True to always write the same
    filename instead (useful when only the latest run matters).
    """
    debug_img = img.copy()
    ImageDraw.Draw(debug_img).line([(x0, y), (x1, y)], fill="red", width=2)
    if overwrite:
        debug_img.save(folder / f"{name}.png")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_img.save(folder / f"{name}_{timestamp}.png")


# --------------------------------------------------------------------------
# Background removal / foot mask
# --------------------------------------------------------------------------

def segment_foot(image_bytes: bytes, session=None, alpha_matting: bool = False) -> tuple[Image.Image, np.ndarray]:
    """
    Remove the background and return (PIL image, alpha channel array).

    Pass a pre-built rembg `session` (created once and reused) when
    possible -- creating a new session reloads the model from disk and
    adds noticeable latency per call. `alpha_matting` is an alternative
    used where a session isn't set up; the two are not combined here
    since neither script needed both.
    """
    if session is not None:
        img_no_bg = remove(image_bytes, session=session)
    else:
        img_no_bg = remove(image_bytes, alpha_matting=alpha_matting)
    img = Image.open(io.BytesIO(img_no_bg))
    alpha = np.array(img)[:, :, 3]
    return img, alpha


def foot_pixel_count(alpha: np.ndarray, threshold: int) -> int:
    """Number of pixels considered part of the foot (alpha above threshold)."""
    return int(np.count_nonzero(alpha > threshold))


def foot_column_margins(alpha: np.ndarray, threshold: int) -> tuple[int, int]:
    """Left-most and right-most columns containing foot pixels, for debug drawing."""
    mask = alpha > threshold
    columns = np.any(mask, axis=0)
    left_margin = int(np.argmax(columns))
    right_margin = int(np.where(columns)[0][-1])
    return left_margin, right_margin


# --------------------------------------------------------------------------
# Axis configuration + generic alignment loop
# --------------------------------------------------------------------------

@dataclass
class AxisConfig:
    """Everything that differs between alignment passes (vertical bridge,
    horizontal bridge, vertical toe, ...)."""

    name: str
    camera_position: str
    laser_to_sensor_offset: int
    laser_offset_sign: int  # +1 if sensor_row = laser_row + offset, -1 if sensor_row = laser_row - offset
    step_size_px: float
    laser_detect_fn: Callable[[Path], Optional[int]]
    direction_positive: object  # enum member used when distance_motor >= 0
    direction_negative: object  # enum member used when distance_motor < 0
    pixel_threshold: int  # how close (px) is "close enough" to stop

    stall_delta: int = 1  # stop if measure barely changed vs. last iteration (avoid thrashing/infinite loop)
    min_pixels_for_valid_foot: Optional[int] = None  # sanity check on segmentation; None to skip
    foot_alpha_threshold: int = 128  # alpha threshold used for the pixel-count sanity check
    mask_alpha_threshold: int = 200  # alpha threshold used to build the foot mask for target-row detection
    min_steps_boost: int = 0  # bump tiny step counts up so the motor visibly moves
    max_iterations: Optional[int] = None
    stop_condition: Optional[Callable[[Motor], bool]] = None  # e.g. limit switch pressed
    confirm_before_move: bool = True  # ask for Enter before each motor.move()
    capture_retries: int = 20

    rembg_session: Optional[object] = None  # pre-built session, reused across calls
    use_alpha_matting: bool = False  # used only if rembg_session is None

    debug_image_name: Optional[str] = None  # if set, save a marked debug image each time a target row is found
    debug_image_overwrite: bool = False  # overwrite the same debug filename instead of timestamping each one


def detect_target_row_with_retries(
    file: Path,
    config: AxisConfig,
    find_target_row: Callable[[np.ndarray], Optional[int]],
    debug_folder: Path,
    max_attempts: int = 10,
) -> Optional[int]:
    """
    Run background removal + target-row detection, retrying on a fresh
    capture if the foot mask is too small/empty or no target row is found.
    """
    current_file = file

    for _ in range(max_attempts):
        image_bytes = current_file.read_bytes()
        img, alpha = segment_foot(image_bytes, session=config.rembg_session, alpha_matting=config.use_alpha_matting)

        if config.min_pixels_for_valid_foot is not None:
            count = foot_pixel_count(alpha, threshold=config.foot_alpha_threshold)
            if count < config.min_pixels_for_valid_foot:
                logger.error("Background removal left too few pixels: %d", count)
                print("O pe esta desfocado na imagem. Ajuste a posicao do pe, mudando-a ligeiramente.")
                time.sleep(3)
                current_file = retry(capture_save, config.camera_position, debug_folder, retries=5)
                if current_file is None:
                    return None
                continue

        row = find_target_row(alpha)
        if row is None:
            logger.error("Target row not detected for %s: trying again", config.name)
            current_file = retry(capture_save, config.camera_position, retries=5)
            if current_file is None:
                return None
            continue

        if config.debug_image_name:
            left, right = foot_column_margins(alpha, threshold=config.mask_alpha_threshold)
            save_debug_line(
                img, left, right, row, debug_folder,
                config.debug_image_name, overwrite=config.debug_image_overwrite,
            )

        return row

    return None


def align_axis(
    motor: Motor,
    config: AxisConfig,
    find_target_row: Callable[[np.ndarray], Optional[int]],
    debug_folder: Path,
) -> None:
    """
    Repeatedly capture, measure, and move until the sensor is within
    `config.pixel_threshold` of the target row, movement stalls, or the
    iteration cap is hit.

    `find_target_row` is called only on the first iteration; after that
    the target is assumed constant, since once the sensor is over the
    foot the background remover can no longer isolate it.
    """
    logger.info("Starting %s alignment", config.name)

    measure = float("inf")
    measure_prev = 0.0
    target_row: Optional[int] = None
    iteration = 1

    while measure > config.pixel_threshold:
        if config.max_iterations and iteration > config.max_iterations:
            logger.info("%s alignment: reached max iterations, stopping", config.name)
            break
        if config.stop_condition and config.stop_condition(motor):
            logger.info("%s alignment: stop condition met, stopping", config.name)
            break

        file = retry(capture_save, config.camera_position, debug_folder, retries=config.capture_retries)
        if file is None:
            logger.error("Could not capture frame for %s alignment.", config.name)
            raise CaptureFailedError(f"Failed to capture frame during {config.name} alignment")

        if iteration == 1:
            target_row = detect_target_row_with_retries(file, config, find_target_row, debug_folder)
            if target_row is None:
                logger.info("Could not detect target row for %s alignment. Aborting pass.", config.name)
                break

        laser_row = retry(config.laser_detect_fn, file)
        if laser_row is None:
            logger.error("Laser dot not detected in image after retries.")
            raise LaserNotDetectedError(f"Laser dot not detected during {config.name} alignment")

        sensor_row = laser_row + config.laser_offset_sign * config.laser_to_sensor_offset
        distance_motor = target_row - sensor_row
        measure = abs(distance_motor)

        logger.info(
            "%s alignment: target_row=%d sensor_row=%d distance=%d measure=%.1f",
            config.name, target_row, sensor_row, distance_motor, measure,
        )

        if measure <= config.pixel_threshold:
            break

        # Stall detection: if the distance barely changed since last iteration,
        # further steps would be negligible (or we're oscillating) -- stop
        # instead of looping indefinitely. Compared against this axis's own
        # previous measure (not another axis's), since align_axis is now
        # shared across axes and their measures are local to this call.
        if iteration > 1 and abs(measure - measure_prev) < config.stall_delta:
            logger.info("%s alignment: movement stalled (delta < %d px), stopping", config.name, config.stall_delta)
            break

        n_steps = int(abs(distance_motor) / config.step_size_px)
        if config.min_steps_boost and n_steps < 4:
            n_steps += config.min_steps_boost

        direction = config.direction_positive if distance_motor >= 0 else config.direction_negative

        if config.confirm_before_move:
            input("Sensor prestes a mexer. Pressione Enter para continuar.")

        logger.info("Moving %s motor. Direction=%s Steps=%d", config.name, direction.name, n_steps)
        try:
            motor.move(direction, n_steps)
            logger.info("%s motor movement completed", config.name)
        except Exception:
            logger.exception("%s positioning failed", config.name)

        measure_prev = measure
        iteration += 1
