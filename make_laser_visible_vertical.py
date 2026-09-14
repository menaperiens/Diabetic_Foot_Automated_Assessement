"""
Sweep the vertical sensor up/down until the alignment laser dot becomes
visible to the camera, or time out.

Used as the recovery step in the pipeline when a detection stage reports
it couldn't see the laser (see alignment_common.EXIT_LASER_NOT_DETECTED
and main_refactored.detect_with_laser_retry).
"""

from __future__ import annotations

import logging
import sys
import time
from time import sleep

from alignment_common import (
    CaptureFailedError,
    LaserNotDetectedError,
    get_logger,
    make_debug_folder,
)
from utils import Motor, VerticalDirection, capture_save, laser_dot_detec_vrt, retry

logger = get_logger(__name__, level=logging.INFO)

DETECTION_TIMEOUT_S = 120  # overall search budget

# Number of detection attempts to make, at the current motor position,
# before moving the motor further in the current direction. Direction
# itself only flips when motor.move() reports a limit switch was hit --
# it is not tied to this attempt count.
ATTEMPTS_PER_MOVE = 2
STEPS_PER_MOVE = 40  # motor steps taken each time ATTEMPTS_PER_MOVE is reached

# A single motor.move() failure is more likely to be a transient GPIO
# hiccup than a real fault, so it's tolerated -- the loop just tries the
# next detection attempt at the same position, direction unchanged. But
# if it fails this many times in a row, that's no longer "transient":
# raise immediately rather than spending the rest of DETECTION_TIMEOUT_S
# retrying a motor that isn't working, only to report the misleading
# "laser not found" at the end instead of the real cause.
MAX_CONSECUTIVE_MOTOR_FAILURES = 3


def main() -> None:
    logger.info("Starting making laser dot visible script")
    debug_folder = make_debug_folder("make_laser_visible_vertical")

    print("A procura do sensor na imagem. Ainda nao encontrado...")

    motor = Motor()
    direction = VerticalDirection.DOWN

    best_dot = None
    count = 0
    consecutive_motor_failures = 0
    start = time.time()

    try:
        while best_dot is None and (time.time() - start) < DETECTION_TIMEOUT_S:
            count += 1
            logger.info("Laser detection attempt %d", count)
            sleep(2)

            file = retry(capture_save, "left", debug_folder)
            if file is None:
                logger.error("Failed to capture image after multiple attempts")
                raise CaptureFailedError("Failed to capture image while searching for laser")

            best_dot = laser_dot_detec_vrt(file)

            if best_dot is None:
                if count % ATTEMPTS_PER_MOVE == 0:
                    old_direction = direction
                    try:
                        direction = motor.move(direction, STEPS_PER_MOVE)
                        consecutive_motor_failures = 0
                        if direction != old_direction:
                            logger.info("Direction changed from %s to %s", old_direction.value, direction.value)
                    except Exception:
                        consecutive_motor_failures += 1
                        logger.exception(
                            "Motor control failed (%d/%d consecutive failures)",
                            consecutive_motor_failures, MAX_CONSECUTIVE_MOTOR_FAILURES,
                        )
                        if consecutive_motor_failures >= MAX_CONSECUTIVE_MOTOR_FAILURES:
                            raise
            else:
                logger.info("Sensor detetado.")
                print("Sensor detetado.")

        logger.info("number of laser detection attemps: %d", count)

        if best_dot is not None:
            logger.info("time taken to detect laser dot: %.2f s", time.time() - start)
        else:
            logger.error("Failed to detect laser dot within %ds", DETECTION_TIMEOUT_S)
            raise LaserNotDetectedError(f"Failed to detect laser dot within {DETECTION_TIMEOUT_S}s")

    except (CaptureFailedError, LaserNotDetectedError) as exc:
        logger.exception("Laser visibility search aborted")
        sys.exit(exc.exit_code)
    finally:
        motor.close()


if __name__ == "__main__":
    main()
