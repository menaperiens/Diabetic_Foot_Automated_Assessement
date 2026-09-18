"""
Offline viewer/animator for MLX90640 thermal captures recorded by the
Arduino streamer + PC receiver pair.

Reads a raw frame .bin file (one 834-uint16 record per subpage, as written
by thermal_receiver.py) and the calibration .bin captured at boot, pairs
consecutive subpages back into full 32x24 temperature images using the
Adafruit MLX90640 driver's conversion math, and animates the result.

DEPENDENCIES
    adafruit_mlx90640.py must sit in the same folder as this script.

CALIBRATION REUSE (OfflineMLX)
    The Adafruit driver is written to read calibration itself over I2C in
    its own __init__. There is no public entry point for "I already have the
    EEPROM dump on disk, just use it" -- so OfflineMLX below skips the
    parent __init__ entirely and pokes the driver's private module-level
    state directly (a.eeData) before calling its private parameter-
    extraction method. This works today against the Adafruit CircuitPython
    driver, but every name it touches is underscore-prefixed, meaning the
    library makes no compatibility promise about them. An upgrade of
    adafruit_mlx90640 could rename or restructure any of this without
    warning; if temperature output starts looking wrong after updating the
    library, this class is the first place to check.
"""

import struct
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Put adafruit_mlx90640.py in the same folder as this script.
# Aliased (not single-letter) so call sites stay readable: mlx_lib.eeData,
# mlx_lib.OPENAIR_TA_SHIFT, etc.
import adafruit_mlx90640 as mlx_lib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAPTURE_DIR = Path("C:/Users/joaos/Desktop/arduino_IR_final/ino_raw_data_bin")
FRAME_FILE = CAPTURE_DIR / "bin" / "2026-09-18_11_44_06_frames_thermal.bin" 
CALIBRATION_FILE = CAPTURE_DIR / "2026-09-17_22_17_53_EE_calb_data.bin"

# One line of comma-separated raw temperatures (deg C) per reconstructed
# frame -- i.e. per subpage PAIR, not per record in the .bin file. Line N
# corresponds to the frame printed as "FRAME NUMBER: N" in the console.
# Named from FRAME_FILE so it's easy to tell which capture it came from.
TEMPERATURES_TXT_FILE = CAPTURE_DIR.parent / "ino_temperatures" / (FRAME_FILE.stem + "_temperatures.txt")


# Words per record. Matches the Arduino sketch's MLX90640Frame buffer: 768
# pixel words (32x24) + 64 auxiliary/housekeeping words + control register +
# subpage number. See thermal_streamer.ino for the full layout.
FRAME_WORDS = 834
FRAME_BYTES = FRAME_WORDS * 2

FRAME_ROWS = 24
FRAME_COLS = 32
PIXEL_COUNT = FRAME_ROWS * FRAME_COLS

# Index of the subpage flag within a frame record (0 or 1). Set by the
# sensor, passed through unchanged by the Arduino sketch.
SUBPAGE_INDEX = 833


EMISSIVITY = 0.95

# Display-only remapping: raw temperatures (deg C, roughly) are rescaled
# into this range before feeding the 'hsv' colormap, purely so the colour
# wheel spans a usable band rather than being dominated by outliers.
# Left as 'hsv', to compare with a previous script that does the live visualization
# and using 'hsv' ; however, a perceptually-ordered map such as
# 'inferno' or 'turbo' would avoid the wraparound where the hottest and
# coldest pixels can end up visually similar colours.
DISPLAY_MIN = 160
DISPLAY_MAX = 360
COLORMAP = "hsv"

ANIMATION_INTERVAL_MS = 100


# ---------------------------------------------------------------------------
# Raw file I/O
# ---------------------------------------------------------------------------


def read_u16_file(path: Path) -> list[int]:
    """Read an entire file as a flat list of little-endian uint16 values."""
    data = path.read_bytes()
    if len(data) % 2 != 0:
        raise ValueError(f"{path.name}: odd byte count, not a uint16 file")
    count = len(data) // 2
    return list(struct.unpack("<" + "H" * count, data))


def count_frames(path: Path) -> int:
    """Number of fixed-size frame records stored in `path`."""
    return path.stat().st_size // FRAME_BYTES


def read_frame(path: Path, frame_index: int) -> list[int]:
    """
    Read one 834-word frame record by index, without loading the rest of
    the file.

    Uses np.fromfile with offset/count to read exactly the bytes needed --
    that part is why this is fast on a large capture, versus re-reading the
    whole file on every call.

    Returns a plain list of Python ints, NOT a numpy array. This matters:
    the Adafruit driver's conversion math (_GetVdd and friends) sign-extends
    raw register values with plain arithmetic like `vdd -= 65536`, relying
    on Python ints having no fixed width. A numpy uint16 scalar does have a
    fixed width and raises OverflowError on that same subtraction. Handing
    the driver a numpy array here reproduces the crash; .tolist() avoids it
    while keeping the fast, bounded disk read.
    """
    offset_bytes = frame_index * FRAME_BYTES
    frame = np.fromfile(path, dtype="<u2", count=FRAME_WORDS, offset=offset_bytes)
    if frame.size != FRAME_WORDS:
        raise IndexError(f"frame {frame_index} out of range in {path.name}")
    return frame.tolist()


# ---------------------------------------------------------------------------
# Subpage pairing
# ---------------------------------------------------------------------------


def next_subpage_pair(frame_path: Path, start_index: int):
    """
    Return the next two frame records that form a valid subpage pair
    starting at `start_index`, plus the index the pair started at.

    The MLX90640 alternates subpages 0/1 between consecutive records under
    normal operation. If a record was ever dropped somewhere upstream (on
    the Arduino's TCP link, for instance), two consecutive records can carry
    the SAME subpage number; pairing them anyway would silently combine two
    reads of one half-image and reconstruct a corrupted frame. When that
    happens this slides forward one record at a time, re-reading BOTH
    records on every attempt so the pair being compared is always current,
    until an alternating pair is found -- and reports how many records had
    to be skipped to get there.

    (Earlier version of this function only re-read the first record of the
    pair inside the retry loop; the second stayed pinned to its original
    read from before the loop started, so it was comparing a moving record
    against a frozen one and could not actually detect a repeat correctly.)

    Returns (subpage_a, subpage_b, pair_index, skipped).
    """
    j = start_index
    subpage_a = read_frame(frame_path, j)
    subpage_b = read_frame(frame_path, j + 1)
    skipped = 0

    while subpage_b[SUBPAGE_INDEX] == subpage_a[SUBPAGE_INDEX]:
        j += 1
        skipped += 1
        subpage_a = read_frame(frame_path, j)
        subpage_b = read_frame(frame_path, j + 1)

    return subpage_a, subpage_b, j, skipped


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class OfflineMLX(mlx_lib.MLX90640):
    """
    An MLX90640 driver instance primed from a saved EEPROM dump instead of
    a live sensor. See the module docstring for why this reaches into the
    driver's private state.
    """

    def __init__(self, ee_words: list[int]):
        # Deliberately NOT calling super().__init__() -- it opens an I2C bus,
        # which does not exist here.
        mlx_lib.eeData[:] = ee_words[:]
        self._ExtractParameters()


# ---------------------------------------------------------------------------
# Frame -> temperature conversion
# ---------------------------------------------------------------------------


def frame_to_temperatures(mlx: OfflineMLX, subpage_a, subpage_b) -> list[float]:
    """
    Run both subpages of one pair through the driver's conversion math and
    return one full 768-pixel temperature list.

    Mirrors the Adafruit driver's own getFrame(): ambient temperature (Ta)
    is read per subpage, and reflected temperature (Tr) is approximated as
    Ta minus a fixed open-air offset rather than measured directly -- the
    same approximation the library itself uses when no reflected-temperature
    sensor is present.
    """
    temps = [0.0] * PIXEL_COUNT

    for subpage in (subpage_a, subpage_b):
        ambient_temp = mlx._GetTa(subpage)
        reflected_temp = ambient_temp - mlx_lib.OPENAIR_TA_SHIFT
        mlx._CalculateTo(subpage, EMISSIVITY, reflected_temp, temps)

    return temps


def temperatures_to_display_values(temps: list[float]) -> np.ndarray:
    """
    Rescale raw temperatures into the fixed DISPLAY_MIN..DISPLAY_MAX band
    used for the colormap, clipped to that band. Purely cosmetic -- has no
    bearing on the temperature values themselves, only on how they are
    coloured.
    """
    temps_arr = np.asarray(temps, dtype=float)
    lo = float(temps_arr.min())
    hi = float(temps_arr.max())
    return np.clip(
        np.interp(temps_arr, [lo, hi], [DISPLAY_MIN, DISPLAY_MAX]),
        DISPLAY_MIN,
        DISPLAY_MAX,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_temperatures_line(f, temps: list[float]) -> None:
    """Append one frame's raw temperatures as a comma-separated line."""
    f.write(",".join(f"{t:.3f}" for t in temps))
    f.write("\n")


def main() -> None:
    ee_words = read_u16_file(CALIBRATION_FILE)
    mlx = OfflineMLX(ee_words)

    total_records = count_frames(FRAME_FILE)

    fig, ax = plt.subplots()
    artists = []

    image_number = 0
    j = 0
    total_skipped = 0

    # Opened once for the whole run and flushed after every line, so the
    # file is up to date and readable even if the animation is closed early
    # or a later frame raises. "w" starts a fresh file per run; switch to
    # "a" if you'd rather append across multiple runs against the same
    # capture.
    with open(TEMPERATURES_TXT_FILE, "w", encoding="utf-8") as temps_file:

        while j < total_records - 1:
            subpage_a, subpage_b, j, skipped = next_subpage_pair(FRAME_FILE, j)
            total_skipped += skipped

            image_number += 1
            print(f"FRAME NUMBER: {image_number}")
            if skipped:
                print(f"  (skipped {skipped} record(s) with a repeated subpage)")

            temps = frame_to_temperatures(mlx, subpage_a, subpage_b)

            write_temperatures_line(temps_file, temps)
            temps_file.flush()

            display_values = temperatures_to_display_values(temps)
            image_matrix = display_values.reshape(FRAME_ROWS, FRAME_COLS)

            im = ax.imshow(
                image_matrix,
                animated=True,
                cmap=COLORMAP,
                vmin=DISPLAY_MIN,
                vmax=DISPLAY_MAX,
            )
            artists.append([im])  # ArtistAnimation expects a list of artists per frame

            j += 1

    print(f"\nTemperatures written to {TEMPERATURES_TXT_FILE}")
    if total_skipped:
        print(f"Total records skipped due to repeated subpages: {total_skipped}")

    ani = animation.ArtistAnimation(
        fig, artists, interval=ANIMATION_INTERVAL_MS, blit=True, repeat=False
    )
    plt.colorbar(artists[0][0], ax=ax, label="Temperature (display units)")
    plt.title("Thermal image")

    def stop_animation(event):
        if event.key == "enter":
            ani.event_source.stop()
            print("Animation stopped")

    fig.canvas.mpl_connect("key_press_event", stop_animation)
    plt.show()


if __name__ == "__main__":
    main()
