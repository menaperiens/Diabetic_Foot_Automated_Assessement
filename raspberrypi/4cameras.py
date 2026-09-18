"""
Capture one image from each of the four cameras (left, right, top, down)
and save them to the final images directory.

Each camera is retried independently; if a camera still fails after all
retries, it's logged and skipped rather than aborting the whole capture --
the pipeline calls this stage with check=False (see
main_refactored.run_stage) specifically so one bad camera doesn't kill
the scan. This script still exits with EXIT_CAPTURE_FAILED if any camera
ultimately failed, so the failure is visible to anything that does
inspect the subprocess result, even though the pipeline currently
doesn't.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import sleep

from alignment_common import EXIT_CAPTURE_FAILED, get_logger
from utils import capture_save

logger = get_logger(__name__, level=logging.INFO)

CAMERA_NAMES = ["left", "right", "top", "down"]
IMAGES_DIR = Path.home() / "Documents" / "images"
MAX_RETRIES_PER_CAMERA = 3
POST_CAPTURE_PAUSE_S = 1  # brief pause after a successful capture before moving to the next camera


def capture_all() -> list[str]:
    """Capture from every camera in CAMERA_NAMES. Returns the names of any cameras that failed."""
    failed_cameras: list[str] = []

    for camera in CAMERA_NAMES:
        for attempt in range(1, MAX_RETRIES_PER_CAMERA + 1):
            try:
                capture_save(camera, IMAGES_DIR)
                sleep(POST_CAPTURE_PAUSE_S)
                logger.info("Successfully saved image from %s camera. attempts=%d", camera, attempt)
                break
            except Exception as e:
                logger.info(
                    "Attempt %d/%d at capturing image from %s camera failed: %s",
                    attempt, MAX_RETRIES_PER_CAMERA, camera, e,
                )
                if attempt == MAX_RETRIES_PER_CAMERA:
                    logger.error("Too many failures trying to capture and save with %s camera. Skipping.", camera)
                    failed_cameras.append(camera)

    return failed_cameras


def main() -> None:
    print("Capturando imagens...")
    failed_cameras = capture_all()
    print("Fim do processo da captura de imagens.")

    if failed_cameras:
        logger.error("Camera capture finished with failures: %s", failed_cameras)
        sys.exit(EXIT_CAPTURE_FAILED)


if __name__ == "__main__":
    main()
