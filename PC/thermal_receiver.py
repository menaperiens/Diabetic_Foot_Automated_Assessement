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

  Calibration goes to one .bin per connection, so a reconnecting
  Arduino does not append a second dump to the first session's files.
  Frames go to one .bin per capture run, named when START is
  received, so a whole run lands in a single pair of files. Each row still
  carries its own frame's arrival time in its first column.
"""

import socket
import struct
import threading
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAVING_DIR = Path("C:/Users/joaos/Desktop/arduino_IR_final/ino_raw_data_bin")

HOST = "0.0.0.0"
DATA_PORT = 8080
TRIGGER_PORT = 8081

TOTAL_FRAMES = 30       # frames captured per START before saving auto-stops
TRIGGER_RECV_BYTES = 1024
CLIENT_DISCONNECT_MSG = "Client disconnected"

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


def append_u16_bin(path: Path, values) -> None:
    with open(path, "ab") as f:
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
        is appended to the same pair of files. Pressing START again begins a
        new run and a new pair.
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

    def record_saved(self) -> None:
        """Count a written frame and disable saving once the quota is met."""
        with self._lock:
            self._saved += 1
            print(f"Saved frame {self._saved}/{self._total}")

            if self._saved >= self._total:
                self._enabled = False
                print(f"Captured {self._total} frames. Saving disabled.")


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


def handle_calibration(values: tuple, stamp: str, session_tag: str) -> None:
    print(f"Received EE packet with {len(values)} words")


    bin_path = SAVING_DIR / "bin" /"EE_calb_data.bin"

    append_u16_bin(bin_path, values)


def handle_frame(values: tuple, state: CaptureState, stamp: str) -> None:
    print("Received FR packet")

    capture_tag = state.capture_tag()
    if capture_tag is None:
        return

    # Named for the capture run, so all frames of one START land together.
    bin_path = SAVING_DIR / "bin" / f"{capture_tag}_frames_thermal.bin"

    # The row keeps this frame's own arrival time, so per-frame timing is
    # still recoverable from inside the file.
    append_u16_bin(bin_path, values)

    state.record_saved()


def serve_client(conn: socket.socket, addr, state: CaptureState) -> None:
    """
    Read packets from one Arduino connection until it goes away.

    The session tag is fixed for the lifetime of this connection, so each
    time the Arduino reconnects its calibration dump lands in a fresh pair of
    files rather than being appended to the previous session's. The board
    re-sends calibration on every reconnect, so without this one file would
    accumulate an extra 832-word row per reconnect, with nothing to indicate
    which frames belong to which row.
    """
    session_tag = now_tag()

    while True:
        packet = read_packet(conn, addr)
        if packet is None:
            break

        # Sampled once, after the packet lands.
        stamp = now_tag()

        tag, values = packet

        if tag == "EE":
            handle_calibration(values, stamp, session_tag)
        elif tag == "FR":
            handle_frame(values, state, stamp)


def main() -> None:
    state = CaptureState(TOTAL_FRAMES)

    threading.Thread(target=trigger_server, args=(state,), daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, DATA_PORT))
        s.listen()

        print(f"Listening on {HOST}:{DATA_PORT} ...")

        while True:
            conn, addr = s.accept()
            print("Connected by", addr)

            with conn:
                serve_client(conn, addr, state)


if __name__ == "__main__":
    main()
