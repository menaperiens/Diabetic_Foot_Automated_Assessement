"""
Receives raw MLX90640 thermal frames from an Arduino over TCP and logs them.

Two servers run side by side:

  port 8080  data     the Arduino connects here and streams packets
  port 8081  trigger  send "START" or "STOP" to control frame capture

WIRE PROTOCOL (Arduino -> PC)

  Each packet is a 2-byte ASCII tag followed by a fixed number of
  little-endian uint16 words. No length prefix, no checksum, no delimiter --
  the sizes below are the only framing there is, so a single desynchronised
  byte corrupts everything after it. Hence the hard disconnect on any
  unrecognised tag.

    "EE" + 832 words   factory calibration EEPROM dump
    "FR" + 834 words   one raw subpage frame
                         [0..767]   pixel words, 32x24 row-major
                         [768..831] auxiliary / housekeeping words
                         [832]      control register 1
                         [833]      subpage number (0 or 1)

OUTPUT

  Both calibration and frames are written as raw .bin files under
  SAVING_DIR / "bin" (created automatically if missing -- see
  ensure_directories()).

  Calibration is written to one FIXED filename, EE_calb_data.bin, and is
  OVERWRITTEN in full on every "EE" packet -- never appended. Since the
  Arduino only ever sends one calibration dump per boot, this file always
  holds exactly the most recent boot's dump, with nothing left over from
  any earlier one.

  Frames go to one .bin per capture run, named when START is received, so
  a whole run lands in a single file. Frame records carry no timestamp of
  their own inside the file -- the run's filename tag is the only timing
  marker, at one-per-run resolution.
"""

import socket
import struct
import threading
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Not hardcoded to a specific user/OS path: Path.home() resolves to the
# current user's home directory on whatever machine this runs on (e.g.
# C:\Users\<name> on Windows, /home/<name> on Linux/macOS). The directory
# itself is created automatically at startup by ensure_directories() below,
# so nothing needs to exist beforehand.
SAVING_DIR = Path.home() / "Documents" / "arduino_IR_final" / "ino_raw_data_bin"

HOST = "0.0.0.0"
DATA_PORT = 8080
TRIGGER_PORT = 8081

TOTAL_FRAMES = 30       # frames captured per START before saving auto-stops
TRIGGER_RECV_BYTES = 1024
CLIENT_DISCONNECT_MSG = "Client disconnected"

CALIBRATION_FILENAME = "EE_calb_data.bin"

# Payload length, in 16-bit words, for each packet tag.
PAYLOAD_WORDS = {"EE": 832, "FR": 834}

HEADER_STRUCT = struct.Struct("2s")

# struct.Struct objects are comparatively expensive to build and there are
# only two payload sizes, so they are built once and reused.
_U16_STRUCT_CACHE: dict[int, struct.Struct] = {}


def u16_struct(count: int) -> struct.Struct:
    """Return a cached little-endian struct for `count` uint16 values."""
    fmt = _U16_STRUCT_CACHE.get(count)
    if fmt is None:
        fmt = struct.Struct("<" + "H" * count)
        _U16_STRUCT_CACHE[count] = fmt
    return fmt


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_tag() -> str:
    """Timestamp string used in filenames. One-second resolution."""
    return datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


def ensure_directories() -> None:
    """
    Create every directory this script writes into, if not already there.

    Called once at startup rather than at import time, so tests (or any
    other caller) that override SAVING_DIR before calling main() still get
    the right directory created, not the default.
    """
    (SAVING_DIR / "bin").mkdir(parents=True, exist_ok=True)


def append_u16_bin(path: Path, values) -> None:
    """Add one record's worth of raw words to the end of `path`."""
    with open(path, "ab") as f:
        f.write(u16_struct(len(values)).pack(*values))


def overwrite_u16_bin(path: Path, values) -> None:
    """Replace `path`'s entire contents with this one record's raw words."""
    with open(path, "wb") as f:
        f.write(u16_struct(len(values)).pack(*values))


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """
    Read exactly n bytes, or return None if the connection closes early.

    A plain recv() may return fewer bytes than asked for, which on a protocol
    with no delimiters would silently shift every subsequent packet.
    """
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Capture state, shared between the trigger thread and the data loop
# ---------------------------------------------------------------------------


class CaptureState:
    """Whether frames are currently being written, and how many so far."""

    def __init__(self, total_frames: int) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._saved = 0
        self._total = total_frames
        self._tag = None

    def start(self) -> None:
        """
        Enable saving, reset the counter, and stamp this capture run.

        The tag is fixed here rather than per packet, so every frame of a run
        is appended to the same file. Pressing START again begins a new run
        and a new file.
        """
        with self._lock:
            self._enabled = True
            self._saved = 0
            self._tag = now_tag()
            print("Saving enabled")

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            print("Saving disabled")

    def capture_tag(self) -> str | None:
        """
        Return the current run's filename tag, or None if saving is off.

        Combines the "should I save?" check with "where do I write?" so both
        are answered under a single acquisition of the lock, and a run that
        ends between the two cannot be written to the wrong file.
        """
        with self._lock:
            return self._tag if self._enabled else None

    def record_saved(self, path: Path) -> None:
        """
        Count a written frame and disable saving once the quota is met.

        Takes the destination path purely so the console output can say
        where the frame went, without a second lock-protected lookup.
        """
        with self._lock:
            self._saved += 1
            print(f"Saved frame {self._saved}/{self._total} -> {path}")

            if self._saved >= self._total:
                self._enabled = False
                print(f"Captured {self._total} frames. Saved to {path}")


# ---------------------------------------------------------------------------
# Trigger server
# ---------------------------------------------------------------------------


def trigger_server(state: CaptureState) -> None:
    """Accept one-shot START/STOP commands on the trigger port, forever."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, TRIGGER_PORT))
        s.listen()

        print(f"Trigger server listening on {TRIGGER_PORT}")

        while True:
            conn, _addr = s.accept()
            with conn:
                command = conn.recv(TRIGGER_RECV_BYTES).decode().strip().upper()

                if command == "START":
                    state.start()
                elif command == "STOP":
                    state.stop()


# ---------------------------------------------------------------------------
# Data server
# ---------------------------------------------------------------------------


def read_packet(conn: socket.socket, addr) -> tuple[str, tuple] | None:
    """
    Read one tagged packet.

    Returns (tag, values), or None if the stream ended or went bad -- in which
    case the caller drops the connection, since the framing cannot be
    recovered once it is lost.
    """
    header = recv_exact(conn, HEADER_STRUCT.size)
    if header is None:
        print(CLIENT_DISCONNECT_MSG, addr)
        return None

    try:
        tag = HEADER_STRUCT.unpack(header)[0].decode("ascii")
    except Exception:
        print("Bad packet header:", header.hex())
        return None

    count = PAYLOAD_WORDS.get(tag)
    if count is None:
        print(f"[DROP] Unknown packet type {tag!r}")
        return None

    payload = recv_exact(conn, count * 2)
    if payload is None:
        print("Client disconnected mid-packet", addr)
        return None

    try:
        values = u16_struct(count).unpack(payload)
    except struct.error as e:
        print(f"[DROP] Bad payload for {tag}: {e}")
        return None

    return tag, values


def handle_calibration(values: tuple) -> None:
    """
    Overwrite the single calibration file with this dump.

    Always a full overwrite, never an append: the Arduino sends exactly one
    dump per boot, so this file is only ever meant to hold the most recent
    one. Appending here would let stale calibration from an earlier boot
    linger indefinitely alongside the current one.
    """
    print(f"Received EE packet with {len(values)} words")

    bin_path = SAVING_DIR / "bin" / CALIBRATION_FILENAME
    overwrite_u16_bin(bin_path, values)

    print(f"Saved calibration to {bin_path}")


def handle_frame(values: tuple, state: CaptureState) -> None:
    print("Received FR packet")

    capture_tag = state.capture_tag()
    if capture_tag is None:
        return

    # Named for the capture run, so all frames of one START land together.
    bin_path = SAVING_DIR / "bin" / f"{capture_tag}_frames_thermal.bin"

    append_u16_bin(bin_path, values)
    state.record_saved(bin_path)


def serve_client(conn: socket.socket, addr, state: CaptureState) -> None:
    """Read packets from one Arduino connection until it goes away."""
    while True:
        packet = read_packet(conn, addr)
        if packet is None:
            break

        tag, values = packet

        if tag == "EE":
            handle_calibration(values)
        elif tag == "FR":
            handle_frame(values, state)


def main() -> None:
    ensure_directories()

    state = CaptureState(TOTAL_FRAMES)

    threading.Thread(target=trigger_server, args=(state,), daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, DATA_PORT))
        s.listen()

        print(f"Listening on {HOST}:{DATA_PORT} ...")
        print(f"Saving under: {SAVING_DIR / 'bin'}")

        while True:
            conn, addr = s.accept()
            print("Connected by", addr)

            with conn:
                serve_client(conn, addr, state)


if __name__ == "__main__":
    main()
