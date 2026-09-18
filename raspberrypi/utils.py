# For camera
import cv2
import numpy as np
import fcntl

# For move_motor
import lgpio
from gpiozero import Button
from enum import Enum

# For logging & debugging
import datetime
import logging

# General use
import time # just to check how long cameras take
from time import sleep
from pathlib import Path
import os

logging.basicConfig(filename=f'main.log',  level=logging.DEBUG,  format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s')

logger = logging.getLogger(__name__)

debug_folder = Path.home() / "Documents" / "DFAA_debug"
debug_folder.mkdir(parents=True, exist_ok=True)

def capture_save(cam_name: str, save_dir: Path | str | None = debug_folder) -> Path | np.ndarray:
    """
    Capture a single frame from the named camera, and optionally save it to disk.

    Args:
        cam_name: Which camera to use — one of "left", "right", "top", "down".
            Determines both the video device index and the output filename
            suffix (see camera_numbers). "left"/"right" frames are rotated
            180 degrees to correct for how those cameras are mounted.
        save_dir: Directory to save the captured image into. Defaults to
            the module-level debug_folder. Pass None explicitly to skip
            saving entirely (see "Return type" below for what you get back
            in that case).

    Returns:
        - Path to the saved image file, if save_dir is not None (the normal case).
        - The raw captured frame as a numpy ndarray (BGR, as returned by
          cam.read()), if save_dir is None. Nothing is written to disk in
          this case.

        The return type therefore depends on which branch is taken — callers
        must know in advance whether they're calling this in "save" or
        "no-save" mode and handle the result accordingly (a Path and an
        ndarray are not interchangeable).

    Raises:
        ValueError: cam_name is not one of the recognized camera names.
        RuntimeError: the camera device could not be opened, a frame could
            not be read from it, or (when saving) the frame could not be
            written to disk.

    Caution — use with retry():
        retry() treats a None return as "this attempt failed, try again."
        A real ndarray or Path returned here is never None, so retry()
        itself won't misfire. But retry() has no way to tell you which
        return type it got back, so if this function is ever called via
        retry() with save_dir=None, the caller must still know to expect
        an ndarray rather than a Path. As of writing, save_dir=None is not
        used through retry() anywhere in this codebase — keep it that way,
        or make the two modes separate functions (e.g. capture() vs
        capture_save()) if that stops being true.

    Notes:
        Captures are serialized across all cameras via a single global
        lock file (/tmp/camera.lock) — see the comment on lock_path for why.
    """

    camera_numbers = {"left": (2, 1), "right": (4, 3), "top": (0, 2), "down": (6, 4)}

    try:
        n = camera_numbers[cam_name][0]
    except KeyError:
        logger.error(f"Invalid camera name: {cam_name}", stacklevel=2)
        raise ValueError(f"Invalid camera name: {cam_name}")

    logger.debug("Reached point just before lock", stacklevel=2)

    # Single global lock, not per-camera: all cameras share the same USB bus/hub,
    # so concurrent captures across cameras can saturate bandwidth and cause
    # dropped frames or capture errors. A per-camera lock was tried and reverted.
    # In practice captures are also always called sequentially in this codebase,
    # so per-camera locking wouldn't buy any real concurrency anyway.
    lock_path = f"/tmp/camera.lock"  
    sleep(0.03) # tried to remove this line. errors ocurred. (timeout when capturing frame)

    with open(lock_path, "w") as lock:
        logger.debug("Lock file opened", stacklevel=2)


        fcntl.flock(lock, fcntl.LOCK_EX)
        logger.debug("Lock acquired", stacklevel=2)
        sleep(0.3)  # tried to remove this line. errors ocurred.(timeout when capturing frame)

        # --- Opening ---

        start = time.monotonic()
        cam = cv2.VideoCapture(n)
        logger.debug(f"{cam_name} camera opened in {time.monotonic() - start:.2f}s",stacklevel=2)
        sleep(2)  # brief warm-up: on this camera, reading immediately after
            # opening sometimes returns ret=False or a garbage frame
         

        # cam.release() must run whichever way we leave this block — on a clean
        # read, a failed read, or an exception from isOpened()/read() — otherwise
        # the device handle leaks and can cause "camera busy" failures on the
        # next capture. Only `ret`/`frame` need to survive past this block; the
        # camera handle itself does not.
        if cam is None or not cam.isOpened():
                logger.error(f"{cam_name} camera not found", stacklevel=2)
                raise RuntimeError(f"{cam_name} camera not found")

            # --- Reading ---

        start = time.monotonic()
        ret, frame = cam.read()
        logger.debug(f"{cam_name} camera captured frame in {time.monotonic() - start:.2f}s", stacklevel=2)


        if not ret:
            logger.error(f"Failed to capture frame from {cam_name} camera", stacklevel=2)
            raise RuntimeError(f"Failed to capture frame from {cam_name} camera")

        if cam_name == "left" or cam_name=="right":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if save_dir is not None:

            ct = datetime.datetime.now()
            lt = ct.strftime("-%H-%M-%S")
            home_path = Path.home()
            file_name = "cam"+ str(camera_numbers[cam_name][1]) + "_" + str(datetime.date.today()) + lt + ".png"
            save_dir=Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = home_path / save_dir / file_name

            
            success = cv2.imwrite(str(file_path), frame) # Depending on your OpenCV/Python binding version, 
                # imwrite expects a str. Some builds coerce Path fine via os.fspath, others don't
            if not success:
                logger.error(f"Failed to save image from {cam_name} camera to {file_path}", stacklevel=2)
                # Must raise here: without it, we'd fall through and return a Path to a
                # file that was never written, and the caller (often via retry()) would
                # have no way of knowing the save failed.
                raise RuntimeError(f"Failed to save image from {cam_name} camera to {file_path}")

        else:
            logger.info("No saving directory provided; returning frame without saving.")
            try:
                cam.release()
            except Exception as release_err:
                logger.warning(f"Error releasing {cam_name} camera: {release_err}")
            return frame   # numpy ndarray (BGR), not a file path

        sleep(3) # let the camera fully release/settle before the lock is dropped and
          # the next capture_save() call can acquire it
    
    logger.debug("Lock file closed (lock released)", stacklevel=2)
    try:
        cam.release()
    except Exception as release_err:
        logger.warning(f"Error releasing {cam_name} camera: {release_err}")
    return Path(file_path) 


def laser_dot_detec_vrt(file):
    """
    Detect the laser dot in an image from the vertical-axis camera.

    Isolates candidate laser-dot contours from `file` using two HSV filters:
    a bright/low-saturation "white core" (the dot itself) and a broader
    "red aura" (pink/red/brown/purple) around it, matching the halo the
    laser casts on the surrounding surface. Candidate contours are scored
    by how much of a ring around their core overlaps the red aura, after
    being filtered on:
      - area (5–100 px), circularity (axis ratio 1–1.8), and a fixed
        positional bound (cx in [512, 542], cy <= 200) reflecting where the
        laser can physically appear relative to the sensor's limit switches
        and mounting geometry (see inline comment at the cx/cy check). Note
        that rejectng cy > 200 means discarding contours positioned 
        in the inferior part of the image, as the y coordinate increases 
        with the decrease in height.
      - aura coverage across 4 angular sectors around the contour ring;
        contours below 0.5 coverage are rejected.
    All of the above numeric thresholds were tuned empirically for this
    camera/lighting setup, not derived analytically.

    Debug images (aura overlay, all contours, per-contour crops, ring
    overlays) are written to ~/Documents/laser_detection_debug/vertical/
    on every call.

    Args:
        file: Path (str or Path-like) to the image to analyze.

    Returns:
        The row (cy, as a float) of the best-scoring laser-dot candidate,
        or None if the image couldn't be loaded or no contour passed all
        filters.
    """
    
    #logger.info("start of laser detection (vertical)", stacklevel=2) too much verbose (retry)

    folder_debug_las_vrt = os.path.expanduser("~/Documents/laser_detection_debug/vertical") 
    # directory where to store debugging images for vertical laser detection
    Path(folder_debug_las_vrt).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(file)
    if img is None:
        logger.error("Failed to load image: %s", file, stacklevel=2)
        return None

    logger.info("Loaded image: %s", file, stacklevel=2)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. White/bright core: high brightness, low saturation
    white_core = cv2.inRange(
    hsv,
    np.array([0, 0, 245]),  # H irrelevant, S low, V high
    np.array([180, 30, 255])
    )

    # 2. Red/pink aura: red hue, medium/high saturation, reasonably bright
    # pink/red aura
    red1 = cv2.inRange(
        hsv,
        np.array([0, 80, 120]),
        np.array([8, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, 80, 120]),
        np.array([180, 255, 255])
    )

    pink = cv2.inRange(
        hsv,
        np.array([150, 60, 240]), # last one was 120
        np.array([169, 255, 255])
    )
    """
    rosy1 = cv2.inRange(
        hsv,
        np.array([0, 25, 150]),
        np.array([12, 180, 255])
    )

    rosy2 = cv2.inRange(
        hsv,
        np.array([165, 25, 150]),
        np.array([180, 180, 255])
    )
    """ # These might come in handy. The colour of the laser dot is slightly
    # different when comparing the hrz vs vert laser dot, possibly due to the surfaces 
    # where they are reflected having different materials/colours. However, if further tuning
    # of the filter colours is needed, these two filters might be useful to add to the red_aura filter.
    

    # Brownish / reddish brown
    brown_red = cv2.inRange(
        hsv,
        np.array([0, 60, 40]),
        np.array([20, 255, 180])
    )

    # Dark purple
    purple_dark = cv2.inRange(
        hsv,
        np.array([120, 60, 40]),
        np.array([160, 255, 180])
    )
    red_aura = pink | red1 | red2 | brown_red | purple_dark # | rosy1 | rosy2

    # NOTE: Most of the numeric thresholds below (area bounds, axis-ratio range,
    # coverage threshold, morphology kernel sizes) were arrived at through manual
    # trial and error on this specific camera/lighting setup, not derived from
    # first principles. Where vrt and hrz differ, it's because each was tuned
    # independently for its own camera — not necessarily a meaningful physical
    # distinction. 

    overlay = img.copy()
    overlay[red_aura > 0] = (0, 255, 0)
    cv2.imwrite(os.path.join(folder_debug_las_vrt, "overlay.png"), overlay)

    contours, _ = cv2.findContours(white_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info("Found %d candidate contours", len(contours), stacklevel=2)

    debug_contours = img.copy()

    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(folder_debug_las_vrt,"all_contours.png"), debug_contours)
    
    best = None
    best_score = -1

    now = datetime.datetime.now()
    folder = os.path.join(folder_debug_las_vrt, f"debug_contours_{now}")
    os.makedirs(folder, exist_ok=True)

    for idx, c in enumerate(contours):
        
        area = cv2.contourArea(c)

        if area < 5 or area > 100:
            # logger.debug("Rejected contour: area too big or small", stacklevel=2)
            # commented because verbose was too much
            continue

        if len(c) < 5:
            continue
        debug_single_contour = img.copy()

        cv2.drawContours(
            debug_single_contour,      # image to draw on
            [c],        # <-- only this contour
            -1,
            (0, 255, 0),
            2
        )

        cv2.imwrite(
            os.path.join(folder, f"contour_{idx}.png"), 
            debug_single_contour
        )

        ellipse = cv2.fitEllipse(c)
        (cx, cy), (w, h), _ = ellipse
        if cy > 200 or cx > 542 or cx < 512:
        # we know the laser can only be within the metal bar -
        #  512 to 542 px in cx (horizontal direction).
        # Also, the sensor can only move so far in both the up-down and down-up
        #  directions due to the limit switches, 
        # The laser is positioned directly above the sensors, and contact
        #  with the limit switch prevents further movement in the same direction. 
        # Thus, the laser cannot be below the lower limit switch, which is
        #  why contours under these conditions are discarded. 
        # The discard of contours with a vertical coordinate greater than
        #  200 had to be added, because the algorithm was detecting false
        # laser points below 200 px in the foot area (pixels from the foot 
        # often pass through the red_aura filter, which, together with the
        #  detection of some white pixels in the vicinity, contributed to
        #  the identification of false positives).
            continue
        if w == 0 or h == 0:
            #logger.debug("Rejected contour: 0s", stacklevel=2)
            # commented because verbose was too much
            continue

        axis_ratio = max(w, h) / min(w, h)
        
        # Almost circular only
        if not (1 <= axis_ratio <= 1.8): # tried different values.
            # 1.2,1.1,1.01 for min and 1.9,1.7 max.
            # 1,1.8 was the best combination.
            logger.debug(
                "Rejected contour %i: axis ratio %.2f outside range", 
                idx, 
                axis_ratio, 
                stacklevel=2
            )
            continue

        logger.debug("Presently analysing contour %i. Passed first filters.", idx, stacklevel=2)

        core_mask = np.zeros_like(white_core)
        cv2.drawContours(core_mask, [c], -1, 255, -1)
        
        kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) # played around with these values, this was best combination achienved
        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5)) # played around with these values, this was best combination achienved

        inner = cv2.dilate(core_mask, kernel_inner, iterations=1)
        outer = cv2.dilate(core_mask, kernel_outer, iterations=1)

        ring = cv2.subtract(outer, inner)

        ring_overlay = img.copy()
        ring_overlay[ring > 0] = (0, 255, 0)  # Green ring
        cv2.imwrite(os.path.join(folder,f"ring_overlay_{idx}.png"), ring_overlay)
        
        aura_in_ring = cv2.bitwise_and(red_aura, red_aura, mask=ring)
        any_pixel_aura = np.any(aura_in_ring)
        logger.debug("Any redish aura pixels in ring surrounding contour? %s",
                    any_pixel_aura, 
                    stacklevel=2
        )
        if any_pixel_aura:
    
            ys, xs = np.where(ring > 0)

            angles = np.arctan2(ys - cy, xs - cx)
            angles = (angles + 2*np.pi) % (2*np.pi)

            num_sectors = 4
            sector_has_red = []

            for i in range(num_sectors):
                a0 = 2*np.pi*i/num_sectors
                a1 = 2*np.pi*(i+1)/num_sectors

                sector_pixels = (angles >= a0) & (angles < a1)

                sector_xs = xs[sector_pixels]
                sector_ys = ys[sector_pixels]

                if len(sector_xs) == 0:
                    sector_has_red.append(False)
                    continue

                red_count = np.count_nonzero(aura_in_ring[sector_ys, sector_xs]>0)

                sector_has_red.append(red_count >= 1)
                
            coverage = sum(sector_has_red) / num_sectors

            if coverage < 0.5:
                logger.debug(
                    "Rejected contour: coverage %.2f below threshold", 
                    coverage,
                    stacklevel=2
                )
                continue

            score = coverage

            logger.debug(
                        "Contour %d passed all tests. score=%.2f",
                        idx, score, stacklevel=2
                        )
            logger.debug("Contour area=%.2f", area, stacklevel=2)
            logger.debug("Contour axis ratio=%.2f", axis_ratio, stacklevel=2)
            logger.debug("Aura coverage=%.2f", coverage, stacklevel=2)

            if score > best_score:
                logger.debug(
                "New best candidate for laser dot: x=%.1f y=%.1f score=%.2f",
                cx, cy, score, stacklevel=2
                )
                best_score = score
                best = (cx, cy)

        
    if best is not None:
        cx, cy = best
        logger.info(
            "Laser detected at x=%.1f y=%.1f with score=%.2f",
            cx, cy, best_score, stacklevel=2
            )
        return cy
    else:
        logger.warning("No laser dot detected", stacklevel=2)
        return
    
def laser_dot_detec_hrz(file):

    """
    Detect the laser dot in an image from the horizontal-axis camera.

    Same white-core / red-aura HSV approach as laser_dot_detec_vrt, tuned
    independently for this camera's view. Differences from the vertical
    version, all empirically settled rather than analytically derived:
      - Wider area bound (5–200 px vs. 100 px) — the dot appears larger
        in this camera's frame.
      - No positional (cx/cy) bounding filter — unlike the vertical case,
        the area/axis-ratio/coverage filters are sufficient on their own.
      - Lower aura-coverage threshold (0.25 vs. 0.5) to avoid false
        rejections; the red-aura filter may need further tuning here if
        this proves too permissive (also includes extra "rosy" hue ranges
        not used in the vertical version, since the laser's reflected
        color differs slightly by surface/material between the two views).

    Debug images (aura overlay, all contours, per-contour crops, ring
    overlays) are written to ~/Documents/laser_detection_debug/horizontal/
    on every call.

    Args:
        file: Path (str or Path-like) to the image to analyze.

    Returns:
        The row (cy, as a float) of the best-scoring laser-dot candidate,
        or None if the image couldn't be loaded or no contour passed all
        filters.
    """
    
    #logger.info("start of laser detection (horizontal)", stacklevel=2) commented as too much verbose due to function retry

    folder_debug_las_hrz = os.path.expanduser("~/Documents/laser_detection_debug/horizontal") 
    # directory where to store debugging images for vertical laser detection
    Path(folder_debug_las_hrz).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(file)
    if img is None:
        logger.error("Failed to load image: %s", file, stacklevel=2)
        return None

    logger.info("Loaded image: %s", file, stacklevel=2)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. White/bright core: high brightness, low saturation
    white_core = cv2.inRange(
    hsv,
    np.array([0, 0, 245]),      # H irrelevant, S low, V high
    np.array([180, 30, 255])
    )

    # 2. Red/pink aura: red hue, medium/high saturation, reasonably bright
    # pink/red aura
    red1 = cv2.inRange(
        hsv,
        np.array([0, 80, 120]),
        np.array([8, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, 80, 120]),
        np.array([180, 255, 255])
    )

    pink = cv2.inRange(
        hsv,
        np.array([150, 60, 240]), # last one was 120
        np.array([169, 255, 255])
    )
    
    # Pale / rosy red
    rosy1 = cv2.inRange(
        hsv,
        np.array([0, 25, 150]),
        np.array([12, 180, 255])
    )

    rosy2 = cv2.inRange(
        hsv,
        np.array([165, 25, 150]),
        np.array([180, 180, 255])
    )


    # Brownish / reddish brown
    brown_red = cv2.inRange(
        hsv,
        np.array([0, 60, 40]),
        np.array([20, 255, 180])
    )

    red_aura = pink |red1 | red2 |brown_red |rosy1 |rosy2

    # NOTE: Most of the numeric thresholds below (area bounds, axis-ratio range,
    # coverage threshold, morphology kernel sizes) were arrived at through manual
    # trial and error on this specific camera/lighting setup, not derived from
    # first principles. Where vrt and hrz differ, it's because each was tuned
    # independently for its own camera — not necessarily a meaningful physical
    # distinction.

    overlay = img.copy()
    overlay[red_aura > 0] = (0, 255, 0)
    cv2.imwrite(os.path.join(folder_debug_las_hrz, "overlay.png"), overlay)

    contours, _ = cv2.findContours(white_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info("Found %d candidate contours", len(contours), stacklevel=2)

    debug_contours = img.copy()

    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(folder_debug_las_hrz,"all_contours.png"), debug_contours)
    

    best = None
    best_score = -1

    now = datetime.datetime.now()
    folder = os.path.join(folder_debug_las_hrz, f"debug_contours_{now}")
    os.makedirs(folder, exist_ok=True)

    for idx, c in enumerate(contours):


        # Unlike laser_dot_detec_vrt, there's no positional bounding filter (cx/cy range)
        # here — it was never needed for the horizontal camera's setup. The other filters
        # (area, axis ratio, aura coverage) are sufficient on their own for this view.
        
        area = cv2.contourArea(c)

        # tuned empirically alongside the other vrt/hrz
        # filter parameters (area bounds, axis-ratio range, coverage threshold);
        # 200 was simply the value that worked best for this camera/setup after
        # trial and error — no specific geometric reason for the vrt/hrz difference.
        if area < 5 or area > 200:
            # logger.debug("Rejected contour: area too big or small", stacklevel=2)
            continue

        if len(c) < 5:
            continue
        contour = img.copy()

        cv2.drawContours(
            contour,      # image to draw on
            [c],        # <-- only this contour
            -1,
            (0, 255, 0),
            2
        )

        cv2.imwrite(os.path.join(folder, f"contour_{idx}.png"), contour)

        ellipse = cv2.fitEllipse(c)
        (cx, cy), (w, h), _ = ellipse

        if w == 0 or h == 0:
            #logger.debug("Rejected contour: 0s", stacklevel=2)
            continue

        axis_ratio = max(w, h) / min(w, h)
        
        # Almost circular only
        # Same circularity filter and tuning history as laser_dot_detec_vrt
        if not (1 <= axis_ratio <= 1.8):
            logger.debug(
                "Rejected contour %i: axis ratio %.2f outside range", 
                idx, 
                axis_ratio, 
                stacklevel=2
            )
            continue

        logger.debug(
            "Presently analysing contour %i. Passed first filters.",
            idx, 
            stacklevel=2
        )
        
        core_mask = np.zeros_like(white_core)
        cv2.drawContours(core_mask, [c], -1, 255, -1)
        
        kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))

        inner = cv2.dilate(core_mask, kernel_inner, iterations=1)
        outer = cv2.dilate(core_mask, kernel_outer, iterations=1)

        ring = cv2.subtract(outer, inner)

        ring_overlay = img.copy()
        ring_overlay[ring > 0] = (0, 255, 0)  # Green ring
        cv2.imwrite(os.path.join(folder,f"ring_overlay_{idx}.png"), ring_overlay)
        
        aura_in_ring = cv2.bitwise_and(red_aura, red_aura, mask=ring)
        any_pixel_aura = np.any(aura_in_ring)
        logger.debug("Any redish aura pixels in ring surrounding contour? %s",
                    any_pixel_aura, 
                    stacklevel=2
        )
        if any_pixel_aura:
            ys, xs = np.where(ring > 0)

            angles = np.arctan2(ys - cy, xs - cx)
            angles = (angles + 2*np.pi) % (2*np.pi)

            num_sectors = 4
            sector_has_red = []

            for i in range(num_sectors):
                a0 = 2*np.pi*i/num_sectors
                a1 = 2*np.pi*(i+1)/num_sectors

                sector_pixels = (angles >= a0) & (angles < a1)

                sector_xs = xs[sector_pixels]
                sector_ys = ys[sector_pixels]

                if len(sector_xs) == 0:
                    sector_has_red.append(False)
                    continue

                red_count = np.count_nonzero(aura_in_ring[sector_ys, sector_xs] > 0)

                sector_has_red.append(red_count >= 1)
                
            coverage = sum(sector_has_red) / num_sectors

            # should we change to coverage < 0.5? we have 0.25 to avoid false rejections. 
            # red-aura filter might need to be improved in hrz case. looking at the pictures and
            # zooming in on the laser dot might be important
            if coverage < 0.25: 
                logger.debug(
                    "Rejected contour: coverage %.2f below threshold", 
                    coverage, 
                    stacklevel=2
                )
                continue

            score = coverage

            logger.debug(
                        "Contour %d passed all tests. score=%.2f",
                        idx, score, stacklevel=2
                        )
            logger.debug("Contour area=%.2f", area, stacklevel=2)
            logger.debug("Contour axis ratio=%.2f", axis_ratio, stacklevel=2)
            logger.debug("Aura coverage=%.2f", coverage, stacklevel=2)



            if score > best_score:
                logger.debug(
                "New best candidate: x=%.1f y=%.1f score=%.2f",
                cx, cy, score, stacklevel=2
                )
                best_score = score
                best = (cx, cy)

        
    if best is not None:
        cx, cy = best
        logger.info(
            "Laser detected at x=%.1f y=%.1f with score=%.2f",
            cx, cy, best_score, stacklevel=2
            )
        return cy
    else:
        logger.warning("No laser dot detected", stacklevel=2)
        return


def retry(func, *args, retries=20, **kwargs):
    for _ in range(retries):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            logger.debug("Attempt failed for %s: %s", func.__name__, e, stacklevel=2)
        sleep(1)
    logger.error(
        "Could not obtain result from %s after %d trials", 
        func.__name__, 
        retries, 
        stacklevel=2
    )
    return None

########################## MOTOR CONTROL CLASSES ###########################

"""
The next two classes are enumerations defining the valid motor movement directions.

An Enum (enumeration) is a Python class that represents a fixed set of
named, constant values. Instead of using arbitrary integers (e.g., 0 and 1)
to represent directions, enums assign descriptive names to those values,
making the code easier to read, maintain, and less prone to errors.

For example, using:

    VerticalDirection.UP

is more meaningful than:

    0

and prevents invalid values from being passed accidentally. Enums also
allow comparisons using descriptive names rather than numeric literals,
improving code clarity and reducing the use of "magic numbers."

These enums define the valid directions for the motorized sensor positioning
system:
    - VerticalDirection: UP and DOWN.
    - HorizontalDirection: FRONT and BACK.
"""

class VerticalDirection(Enum):
    UP = 0 # to move upwards
    DOWN = 1 # to move downwards

class HorizontalDirection(Enum):
    BACK = 0 # to move sensors backwards, towards the heel
    FRONT = 1 # to move sensors forwards, away from the heel
      

"""The goal of following class is to control the motor. The
parameters to be specified are:
 - the number of the direction and pulse pins corresponding to the vertical 
stepper (dir_h and step_h)
 - the number of the direction and pulse pins corresponding to the
horizontal stepper (dir_v and step_v)
 - the numbers for the two vertical stepper limit switches 
(stp_lim_down and stp_lim_up)
 - the numbers for the two horizontal stepper limit switches
(stp_lim_front and stp_lim_back)
"""

class Motor:

    """
    Controls the two-axis (vertical + horizontal) stepper sensor rig via
    lgpio, with four limit switches (gpiozero Buttons) to stop movement
    safely at each axis's physical extremes.
    """

    def __init__(self, dir_h=13, step_h=26, dir_v=5, step_v=6, 
        stp_lim_down = 22,
        stp_lim_up = 23,
        stp_lim_front = 27,
        stp_lim_back = 4
    ):

        self.chip = lgpio.gpiochip_open(0) # This function opens the first GPIO 
        # chip (chip 0) on the system.
        # The return value, stored in h, is a handle to the GPIO chip. 
        # This handle is required to interact with the chip later 
        # (to claim pins, write data, read data, etc.)

        self.dir_h = dir_h
        self.step_h = step_h
        self.dir_v = dir_v
        self.step_v = step_v

        for pin in (dir_h, step_h, dir_v, step_v):
            lgpio.gpio_claim_output(self.chip, pin)


        self.delay_high = 0.004 # square wave - how long the pulse is high
        self.delay_low = 0.004 # square wave - how long the pulse is low
                            # if we increase to 0.05 the motor will move in jerks
                            # this was chosen by trial and error

        # Steppers are driven directly via lgpio (raw pulse timing control).
        # Limit switches use gpiozero's Button instead of lgpio, since Button
        # gives debouncing and a simple is_pressed check for free.
        self.stp_lim_down = Button(stp_lim_down)
        self.stp_lim_up = Button(stp_lim_up)
        self.stp_lim_front = Button(stp_lim_front)
        self.stp_lim_back = Button(stp_lim_back)

    def move(
        self,
        direction: VerticalDirection | HorizontalDirection,
        n_steps: int = 20,
    ):
        """
        Step the motor in `direction` for up to `n_steps` pulses, stopping
        early if a limit switch is hit.

        Returns:
            - If a limit switch is hit mid-move: the OPPOSITE direction
            (e.g. hits the lower limit while going DOWN -> returns UP),
            as a hint for which way is safe to move next.
            - If all n_steps complete without hitting a limit: the SAME
            `direction` that was passed in.
            Callers that care about limit-hit detection should compare the
            return value against `direction`, not just check truthiness.
        """

        if isinstance(direction, VerticalDirection):
            dir_pin = self.dir_v
            step_pin = self.step_v
            flag_vert = True

        elif isinstance(direction, HorizontalDirection):
            dir_pin = self.dir_h
            step_pin = self.step_h
            flag_vert = False

        else:
            raise TypeError("Invalid direction type")

        lgpio.gpio_write(self.chip, dir_pin, direction.value)

        
        for _ in range(n_steps):
        
            # Stop if limit reached BEFORE stepping
            if flag_vert: # if we are oriented in the vertical direction
    
                # Going DOWN and lower limit reached
                # (direction.value is 1 for DOWN/FRONT and 0 for UP/BACK, so it doubles
                #  as the boolean check here — see VerticalDirection/HorizontalDirection)
                if self.stp_lim_down.is_pressed and direction.value:
                    logger.info("O sensor chegou ao limite inferior.", stacklevel=2)
                    return VerticalDirection.UP
    
                # Going UP and upper limit reached
                if self.stp_lim_up.is_pressed and not direction.value:
                    logger.info("O sensor chegou ao limite superior.", stacklevel=2)
                    return VerticalDirection.DOWN
                
            else: # the following division and consequent return of different outputs dependding on
                # the direction is not needed in the present code
                # but its left here for coherence with code for vertical case and for possible future use
    
                if self.stp_lim_front.is_pressed and direction.value:
                # in the horizontal case, if going towards camera, away from foot and reach limit (default pin 27 is pressed), stop.
                    logger.info("O sensor chegou ao limite horizontal posterior.", stacklevel=2)
                    return HorizontalDirection.BACK
                
                if (self.stp_lim_back.is_pressed and not direction.value):
                # in the horizontal case, if going towards power supply and foot and reach limit (default pin 4 is pressed), stop.
                    logger.info("O sensor chegou ao limite horizontal anterior.", stacklevel=2)
                    return HorizontalDirection.FRONT
    
            lgpio.gpio_write(self.chip, step_pin, 1)
            sleep(self.delay_high)
    
            lgpio.gpio_write(self.chip, step_pin, 0)
            sleep(self.delay_low)
                
        return direction
    
    def stp_lim_getter(self, stp_lim: VerticalDirection | HorizontalDirection = HorizontalDirection.FRONT):
        # Returns the Button object for the given limit switch, e.g.
        # stp_lim_getter(VerticalDirection.UP) -> self.stp_lim_up.
        # Function: to acesss the limit switches from outside the class.
        # Default chosen as it was first limit switch we had to access.
        return getattr(self, f"stp_lim_{stp_lim.name.lower()}")
        
    def close(self):
        lgpio.gpiochip_close(self.chip)
