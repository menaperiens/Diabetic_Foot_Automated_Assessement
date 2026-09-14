"""
Foot/toe scanning rig controller.

Waits for a physical button press (GPIO pin 12), then:
  1. Notifies a PC-side Arduino bridge over TCP that a scan is starting.
  2. Runs a sequence of subprocess stages that home the horizontal sensor,
     make an alignment laser visible, detect the lower foot position,
     detect the toe, detect the upper foot/bridge position, and capture
     images from 4 cameras while a watchdog keeps the Pi/PC link alive.

Each iteration is bounded by an overall timeout (PIPELINE_TIMEOUT_S). If a
stage fails or the timeout is hit, the error is logged and the rig loops
back to waiting for the next button press instead of crashing.
"""

import logging
import socket
import subprocess
from time import sleep

from gpiozero import Button
from stopit import ThreadingTimeout, TimeoutException

# --- Configuration -----------------------------------------------------

BUTTON_PIN = 12
PC_IP = "IP_REMOVIDO"
PC_PORT = PORTA_REMOVIDA
SOCKET_TIMEOUT_S = 5
PIPELINE_TIMEOUT_S = 1000
LASER_RETRY_ATTEMPTS = 20
LASER_NOT_DETECTED_RETURN_CODE = 10

# Set to True only for manual/debug runs where you want to step through
# each stage by hand. Leave False for unattended operation.
INTERACTIVE_DEBUG = False

logging.basicConfig(
    filename="main.log",
    level=logging.DEBUG,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
)

button = Button(BUTTON_PIN)


# --- Helpers -------------------------------------------------------------

def pause_for_debug(message="Press Enter to continue..."):
    """Optional manual checkpoint. No-op unless INTERACTIVE_DEBUG is True."""
    if INTERACTIVE_DEBUG:
        input(message)


def wait_for_button_press():
    """Block until the start button is pressed."""
    print("Press the button to begin.")
    while not button.value:
        sleep(0.5)
    print("Button pressed. Starting process...")


def notify_arduino():
    """Tell the PC-side Arduino bridge to start, over TCP.

    Logs and returns without raising on any connection failure, since a
    failed notification currently should not by itself abort the scan.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(SOCKET_TIMEOUT_S)
            s.connect((PC_IP, PC_PORT))
            s.sendall(b"START")
        logging.info("Sent trigger to Arduino successfully")
    except socket.timeout:
        logging.error("Connection to PC timed out")
    except ConnectionRefusedError:
        logging.error("PC refused the connection (is the server running?)")
    except OSError as e:
        logging.error("Socket error: %s", e)


def run_stage(script_name, check=True):
    """Run one pipeline stage as a subprocess and return the CompletedProcess."""
    return subprocess.run(["python3", script_name], check=check)


def detect_with_laser_retry(detection_script, laser_script):
    """Run a detection stage, retrying if the laser wasn't detected.

    The detection script is expected to exit with
    LASER_NOT_DETECTED_RETURN_CODE when it can't see the laser, in which
    case we re-run the "make laser visible" script and try again.
    Returns the successful CompletedProcess, or None if all attempts failed.
    """
    for attempt in range(1, LASER_RETRY_ATTEMPTS + 1):
        try:
            return run_stage(detection_script, check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == LASER_NOT_DETECTED_RETURN_CODE:
                logging.info(
                    "Laser not detected (attempt %d/%d). Re-running %s...",
                    attempt, LASER_RETRY_ATTEMPTS, laser_script,
                )
                run_stage(laser_script, check=True)
                continue
            logging.error(
                "%s failed with unexpected return code: %d",
                detection_script, e.returncode,
            )
            return None
    logging.error(
        "%s: laser still not detected after %d attempts. Giving up.",
        detection_script, LASER_RETRY_ATTEMPTS,
    )
    return None


def run_pipeline():
    """Run the full scan pipeline once. Raises on unrecoverable stage failure."""
    run_stage("move_sensor_limit_horizontal.py")
    pause_for_debug()

    watchdog = subprocess.Popen(["python3", "pi-win_connection_scheduled.py"])
    try:
        sleep(3)  # give the watchdog time to fully start
        pause_for_debug()

        run_stage("4cameras.py", check=False)
    finally:
        # Always stop the watchdog, even if camera capture raised.
        watchdog.terminate()

    pause_for_debug()

    run_stage("make_laser_visible_vertical_v2.py")
    pause_for_debug()

    detect_with_laser_retry(
        "foot_detect_motor_control_vertical_lower_positioning.py",
        "make_laser_visible_vertical_v2.py",
    )
    pause_for_debug()

    run_stage("toe_detec_mot_cont.py")
    print("Measuring...")
    sleep(5)

    run_stage("move_sensor_limit_horizontal.py")
    pause_for_debug()

    run_stage("make_laser_visible_vertical_v2.py")
    pause_for_debug()

    detect_with_laser_retry(
        "foot_bridge_detect_motor_control.py",
        "make_laser_visible_vertical_v2.py",
    )
    pause_for_debug()

    run_stage("move_sensor_limit_horizontal.py")
    print("Done! Success.")


def main():
    while True:
        logging.info("Starting new iteration (main loop)")
        wait_for_button_press()
        notify_arduino()

        try:
            with ThreadingTimeout(PIPELINE_TIMEOUT_S, swallow_exc=False):
                run_pipeline()
        except TimeoutException:
            logging.error("Pipeline took too long. Aborting.")
            print("Pipeline took too long. Aborting.")
        except subprocess.CalledProcessError as e:
            logging.error(
                "Stage '%s' failed with return code %d. Aborting this run.",
                e.cmd, e.returncode,
            )
            print(f"A stage failed ({e.cmd}). Check main.log for details.")
        except Exception:
            logging.exception("Unexpected error during pipeline run.")
            print("Unexpected error. Check main.log for details.")

        logging.info("Restarting...")
        sleep(2)


if __name__ == "__main__":
    main()
