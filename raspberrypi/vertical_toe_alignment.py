"""
Toe row detection and vertical motor alignment.

Positions the vertical sensor at the row where the toes begin (the
bottom edge of the left third of the foot). See alignment_common.py for
the shared capture -> segment -> find target row -> detect laser -> move
control loop; this file only holds what's specific to the toe target:
how its row is found, and its tuned constants.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from rembg import new_session

from alignment_common import (
    AlignmentError,
    AxisConfig,
    align_axis,
    get_logger,
    make_debug_folder,
)
from utils import Motor, VerticalDirection, laser_dot_detec_vrt

logger = get_logger(__name__, level=logging.DEBUG)


# --------------------------------------------------------------------------
# Constants
#
# Values below were tuned empirically on the physical rig. Do not change
# them unless the camera or sensor mounting has physically moved.
# --------------------------------------------------------------------------

PX_THRESHOLD = 33
STALL_DELTA = 3
MIN_FOOT_PIXELS = 10_000
MAX_ITERATIONS = 10

# Calibration offset between the laser centroid and the sensor's center row.
# Derived empirically from a reference photo: sensor center was at row 348,
# laser centroid was at row 6 -> offset = 348 - 6 = 342 pixels.
LASER_TO_SENSOR_OFFSET = 348 - 6

# The original script used step_size=3 together with a separate `N_STEPS *= 4`
# correction discovered during calibration -- two knobs doing one job, so
# nobody could say what the "real" pixels-per-step figure was. They're
# combined into a single effective value here: 3 steps-per-px / 4 correction
# == 0.75 px per step. If this ever needs recalibrating, tune this one
# number directly instead of reintroducing a second multiplier.
STEP_SIZE_PX = 0.75

FOOT_ALPHA_THRESHOLD = 128
TOE_MASK_ALPHA_THRESHOLD = 200


# --------------------------------------------------------------------------
# Target-row detection (toe-specific)
# --------------------------------------------------------------------------

def find_toe_row(alpha: np.ndarray) -> Optional[int]:
    """
    Target row: the bottom-most row where the foot appears within the
    left third of the image (columns 0 to width/3). This corresponds to
    where the toes begin.
    """
    mask = alpha > TOE_MASK_ALPHA_THRESHOLD
    _, width_img = mask.shape
    left_piece = mask[:, :width_img // 3]
    rows_with_foot = np.where(np.any(left_piece, axis=1))[0]
    if len(rows_with_foot) == 0:
        # The original script indexed [-1] here unconditionally, which
        # raises IndexError if no foot pixels fall in the left third.
        # Returning None instead routes back through the normal
        # "target row not found, retry capture" path.
        return None
    return int(rows_with_foot[-1])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    debug_folder = make_debug_folder("vertical_toe_alignment")
    motor = Motor()
    session = new_session("u2net")  # built once, reused across the pass

    toe_config = AxisConfig(
        name="toe",
        camera_position="left",
        laser_to_sensor_offset=LASER_TO_SENSOR_OFFSET,
        laser_offset_sign=+1,
        step_size_px=STEP_SIZE_PX,
        laser_detect_fn=laser_dot_detec_vrt,
        direction_positive=VerticalDirection.DOWN,
        direction_negative=VerticalDirection.UP,
        pixel_threshold=PX_THRESHOLD,
        stall_delta=STALL_DELTA,
        max_iterations=MAX_ITERATIONS,
        min_pixels_for_valid_foot=MIN_FOOT_PIXELS,
        foot_alpha_threshold=FOOT_ALPHA_THRESHOLD,
        mask_alpha_threshold=TOE_MASK_ALPHA_THRESHOLD,
        rembg_session=session,
        debug_image_name="marked",
        debug_image_overwrite=True,  # original script always wrote marked.png, not one file per run
        confirm_before_move=True,  # was missing in the original toe script; now matches the bridge script
    )

    try:
        align_axis(motor, toe_config, find_toe_row, debug_folder)
    except AlignmentError as exc:
        logger.exception("Toe alignment aborted")
        sys.exit(exc.exit_code)
    finally:
        motor.close()


if __name__ == "__main__":
    main()
