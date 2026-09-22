"""
Foot/toe scanning rig controller.

Waits for a physical button press (GPIO pin 12), then:
  1. Notifies a PC-side Arduino bridge over TCP that a scan is starting.
  2. Runs a sequence of subprocess stages that home the horizontal sensor,
     make an alignment laser visible, detect the lower foot position,
     detect the toe, detect the upper foot/bridge position, and capture
     images from 4 cameras while a watchdog keeps the Pi/PC link alive.

Each iteration is bounded by an overall timeout (PIPELINE_TIMEOUT_S), and
each individual stage is additionally bounded by STAGE_TIMEOUT_S (see the
note on that constant below -- it's a safety net against a wedged
subprocess, not a bound on legitimate human confirmation time). If a
stage fails or a timeout is hit, the error is logged and the rig loops
back to waiting for the next button press instead of crashing.
"""

import logging
import socket
import subprocess
from time import sleep

from gpiozero import Button
from stopit import ThreadingTimeout, TimeoutException

from alignment_common import EXIT_LASER_NOT_DETECTED as LASER_NOT_DETECTED_RETURN_CODE, EXIT_CAPTURE_FAILED 

# --- Configuration -----------------------------------------------------

BUTTON_PIN = 12
PC_IP = "IP_REMOVIDO"
PC_PORT = PORTA_REMOVIDA
SOCKET_TIMEOUT_S = 5
PIPELINE_TIMEOUT_S = 1000

# Last-resort guard against a subprocess that never returns at all -- e.g.
# a camera driver stall inside cv2.VideoCapture.read(), which has no
# internal timeout of its own. This is NOT meant to bound legitimate
# human confirmation time: several stages (move_sensor_limit_horizontal.py,
# the vertical bridge/toe alignment scripts, the interactive toe-marking
# UI in horizontal_toe_alignment.py) prompt with input() or an on-screen
# pygame session before moving, and a technician may legitimately take a
# while to respond. STAGE_TIMEOUT_S is deliberately generous so it only
# fires on a genuinely wedged stage, not a slow human -- tune it down only
# if you've confirmed how long the interactive stages actually take in
# practice, and note PIPELINE_TIMEOUT_S is the tighter overall budget
# that will usually trip first for a stage that's merely slow.
STAGE_TIMEOUT_S = 300

LASER_RETRY_ATTEMPTS = 20

WATCHDOG_STARTUP_S = 3  # time allowed for the transfer watchdog to connect before the first capture
WATCHDOG_SHUTDOWN_S = 10  # time allowed for it to finish an in-flight transfer on shutdown

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
    print("Pressione o botão para iniciar.")
    while not button.value:
        sleep(0.5)
    print("Botão pressionado. Iniciando processo...")


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


def run_stage(script_name, check=True, timeout=STAGE_TIMEOUT_S):
    """
    Run one pipeline stage as a subprocess and return the CompletedProcess.

    `timeout` is enforced by the OS (subprocess.run kills the child on
    expiry) -- unlike ThreadingTimeout around the pipeline as a whole,
    which can't preempt a thread that's blocked inside a subprocess wait,
    this actually stops a wedged stage. Pass timeout=None to disable it
    for a stage that's known to need open-ended human interaction time.
    """
    return subprocess.run(["python3", script_name], check=check, timeout=timeout)


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
    run_stage("move_sensor_limit_horizontal.py", check=False)
    pause_for_debug()

    # The image-transfer watchdog is NOT started here: it runs for the
    # lifetime of the rig process (see main()). Starting and killing it
    # around each capture used to race the transfers it was meant to
    # perform -- terminate() fired as soon as 4cameras.py returned, while
    # the last image was still settling or mid-upload.
    run_stage("4cameras.py", check=False)

    pause_for_debug()

    run_stage("make_laser_visible_vertical.py")
    pause_for_debug()

    detect_with_laser_retry(
        "vertical_toe_alignment.py",
        "make_laser_visible_vertical.py",
    )
    pause_for_debug()

    run_stage("horizontal_toe_alignment.py")

    wait_for_button_press()
    print("Measuring...")
    sleep(5)

    run_stage("move_sensor_limit_horizontal.py")
    pause_for_debug()

    run_stage("make_laser_visible_vertical.py")
    pause_for_debug()

    detect_with_laser_retry(
        "foot_bridge_align.py",
        "make_laser_visible_vertical.py",
    )
    wait_for_button_press()
    pause_for_debug()

    run_stage("move_sensor_limit_horizontal.py")
    print("Done! Success.")


def start_transfer_watchdog():
    """Start the Pi->PC image transfer watcher as a long-lived subprocess."""
    logging.info("Starting image transfer watchdog")
    watchdog = subprocess.Popen(["python3", "pi-win_conn.py"])
    sleep(WATCHDOG_STARTUP_S)  # give it time to connect before the first capture
    return watchdog


def main():
    # Started once, outside the loop: it's a folder watcher, so it has no
    # reason to be scoped to a single scan, and connecting over SSH once at
    # startup beats re-authenticating on every run.
    watchdog = start_transfer_watchdog()

    try:
        run_forever(watchdog)
    finally:
        logging.info("Stopping image transfer watchdog")
        watchdog.terminate()
        try:
            watchdog.wait(timeout=WATCHDOG_SHUTDOWN_S)
        except subprocess.TimeoutExpired:
            logging.warning("Transfer watchdog did not stop in time; killing it.")
            watchdog.kill()


def run_forever(watchdog):
    while True:
        logging.info("Starting new iteration (main loop)")

        # A long-lived subprocess can die (dropped SSH connection, crash).
        # Nothing would restart it otherwise, and images would silently stop
        # reaching the PC, so check before each run.
        if watchdog.poll() is not None:
            logging.error("Transfer watchdog exited with code %s. Restarting it.", watchdog.returncode)
            watchdog = start_transfer_watchdog()

        wait_for_button_press()
        notify_arduino()

        try:
            with ThreadingTimeout(PIPELINE_TIMEOUT_S, swallow_exc=False):
                run_pipeline()
        except TimeoutException:
            logging.error("Pipeline took too long. Aborting.")
            print("Pipeline demorou demasiado tempo. Abortando.")
        except subprocess.TimeoutExpired as e:
            logging.error(
                "Stage '%s' did not finish within %ds and was killed. Aborting this run.",
                e.cmd, e.timeout,
            )
            print(f"Uma etapa do teste demorou demasiado tempo. Abortando.")
        except subprocess.CalledProcessError as e:
            logging.error(
                "Stage '%s' failed with return code %d. Aborting this run.",
                e.cmd, e.returncode,
            )
            if e.returncode == LASER_NOT_DETECTED_RETURN_CODE:
              print("Erro na deteção do laser. Requerida a atenção de um técnico.")
            if e.returncode == EXIT_CAPTURE_FAILED:
              print("Erro relativo às câmaras. Requerida a atenção de um técnico.")
        except Exception:
            logging.exception("Unexpected error during pipeline run.")
            print("Erro inesperado. Requerida a atenção de um técnico.")

        logging.info("Restarting...")
        sleep(2)


if __name__ == "__main__":
    main()
