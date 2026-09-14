"""
Foot bridge (peito do pe) detection and motor alignment.

Positions the vertical and horizontal sensors on top of the foot's
bridge. See alignment_common.py for the shared capture -> segment ->
find target row -> detect laser -> move control loop; this file only
holds what's specific to the bridge: how the target row is found for
each axis, and the per-axis tuned constants.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from rembg import new_session

from alignment_common import (
    LaserError,
    CameraError,
    AxisConfig,
    align_axis,
    get_logger,
    make_debug_folder,
)
from utils import HorizontalDirection, Motor, VerticalDirection, laser_dot_detec_hrz, laser_dot_detec_vrt

logger = get_logger(__name__, level=logging.INFO)


# --------------------------------------------------------------------------
# Constants
#
# Values below were tuned empirically on the physical rig. Do not change
# them unless the camera or sensor mounting has physically moved.
# --------------------------------------------------------------------------

PX_THRESHOLD = 7
STALL_DELTA = 1
MIN_FOOT_PIXELS = 10_000
MAX_HORIZONTAL_ITERATIONS = 10

VERTICAL_LASER_TO_SENSOR_OFFSET = 120  # measured: in one picture, sensor row=210, laser centroid row=3 -> later tuned to 120
VERTICAL_STEP_SIZE_PX = 0.6

HORIZONTAL_LASER_TO_SENSOR_OFFSET = 242 - 22  # measured: in one picture, laser row=242, sensor row=22
HORIZONTAL_STEP_SIZE_PX = 0.5
HORIZONTAL_MIN_STEPS_BOOST = 5

FOOT_ALPHA_THRESHOLD = 128
BRIDGE_ALPHA_THRESHOLD_VERTICAL = 254
BRIDGE_ALPHA_THRESHOLD_HORIZONTAL = 200


# --------------------------------------------------------------------------
# Target-row detection (bridge-specific)
# --------------------------------------------------------------------------

def find_vertical_bridge_row(alpha: np.ndarray) -> Optional[int]:
    """
    Target row for vertical alignment: the top-most row where the foot
    appears within the middle 40%-60% column band. Using a band instead
    of a single column tolerates small horizontal irregularities in the
    detected foot boundary.
    """
    mask = alpha > BRIDGE_ALPHA_THRESHOLD_VERTICAL
    _, width_img = mask.shape
    band = mask[:, width_img * 4 // 10: width_img * 6 // 10]
    rows_with_foot_in_band = np.any(band, axis=1)
    if not rows_with_foot_in_band.any():
        return None
    return int(np.argmax(rows_with_foot_in_band))


def find_horizontal_bridge_row(alpha: np.ndarray) -> Optional[int]:
    """
    Target row for horizontal alignment: one third of the way down the
    foot's row extent.
    """
    mask = alpha > BRIDGE_ALPHA_THRESHOLD_HORIZONTAL
    foot_rows = np.where(np.any(mask, axis=1))[0]
    if len(foot_rows) == 0:
        return None
    bridge_row = int((foot_rows[0] + foot_rows[-1]) // 3)
    logger.info("Horizontal bridge_row: %d", bridge_row)
    return bridge_row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    debug_folder = make_debug_folder("foot_bridge_align")
    motor = Motor()
    session = new_session("u2net")  # built once, reused across the vertical pass

    vertical_config = AxisConfig(
        name="vertical",
        camera_position="left",
        laser_to_sensor_offset=VERTICAL_LASER_TO_SENSOR_OFFSET,
        laser_offset_sign=+1,
        step_size_px=VERTICAL_STEP_SIZE_PX,
        laser_detect_fn=laser_dot_detec_vrt,
        direction_positive=VerticalDirection.DOWN,
        direction_negative=VerticalDirection.UP,
        pixel_threshold=PX_THRESHOLD,
        stall_delta=STALL_DELTA,
        min_pixels_for_valid_foot=MIN_FOOT_PIXELS,
        foot_alpha_threshold=FOOT_ALPHA_THRESHOLD,
        mask_alpha_threshold=BRIDGE_ALPHA_THRESHOLD_VERTICAL,
        rembg_session=session,
        debug_image_name="bridge_position_(vertical_row)",
    )

    horizontal_config = AxisConfig(
        name="horizontal",
        camera_position="top",
        laser_to_sensor_offset=HORIZONTAL_LASER_TO_SENSOR_OFFSET,
        laser_offset_sign=-1,
        step_size_px=HORIZONTAL_STEP_SIZE_PX,
        laser_detect_fn=laser_dot_detec_hrz,
        direction_positive=HorizontalDirection.BACK,
        direction_negative=HorizontalDirection.FRONT,
        pixel_threshold=PX_THRESHOLD,
        stall_delta=STALL_DELTA,
        mask_alpha_threshold=BRIDGE_ALPHA_THRESHOLD_HORIZONTAL,
        min_steps_boost=HORIZONTAL_MIN_STEPS_BOOST,
        max_iterations=MAX_HORIZONTAL_ITERATIONS,
        stop_condition=lambda m: m.stp_lim_back.is_pressed,
        use_alpha_matting=True,
    )

    try:
        align_axis(motor, vertical_config, find_vertical_bridge_row, debug_folder)
        align_axis(motor, horizontal_config, find_horizontal_bridge_row, debug_folder)
    except CameraError:
        logger.exception("Alignment aborted")
        sys.exit(11)
    except LaserError:
        logger.exception("Alignment aborted")
        sys.exit(10)
    finally:
        motor.close()


if __name__ == "__main__":
    main()
