"""
Move the horizontal sensor away from the foot until it reaches its
front limit switch (homing), or time out.
"""

from __future__ import annotations

import logging
import sys
import time

from alignment_common import EXIT_MOVEMENT_FAILED, get_logger
from utils import HorizontalDirection, Motor

logger = get_logger(__name__, level=logging.INFO)

TIMEOUT_SECONDS = 15
STEP_BATCH = 10  # small batch so we can keep checking elapsed time between moves


def main() -> None:
    input("Processo de afastamento do sensor do pe iniciado. Presse Enter para mover.")

    motor = Motor()
    # Pass the direction explicitly rather than relying on stp_lim_getter's
    # default (which today happens to be FRONT) -- otherwise this script's
    # correctness would silently depend on a default defined elsewhere.
    limit = motor.stp_lim_getter(HorizontalDirection.FRONT)

    try:
        start = time.monotonic()
        logger.info("Starting horizontal movement away from foot.")

        while not limit.is_pressed and time.monotonic() - start < TIMEOUT_SECONDS:
            motor.move(direction=HorizontalDirection.FRONT, n_steps=STEP_BATCH)

        if not limit.is_pressed:
            logger.error("Sensor demorou demasiado tempo a atingir ponto de chegada.")
            print("Ocorreu um erro durante o processo de afastamento do sensor do pe.")
            # A timed-out homing move is a real operational failure, not just
            # something to log -- exit non-zero so a caller (script, wrapper,
            # orchestrator) can detect it instead of it looking like success.
            sys.exit(EXIT_MOVEMENT_FAILED)

        logger.info("Horizontal end-stop reached successfully")
        print("Processo de afastamento do sensor do pe terminado.")

    except Exception:
        logger.exception("Horizontal movement failed")
        print("Ocorreu um erro durante o processo de afastamento do sensor do pe.")
        raise
    finally:
        logger.info("Closing GPIO chip")
        motor.close()


if __name__ == "__main__":
    main()
