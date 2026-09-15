"""
Watch the Pi's images folder and copy each new image to the Windows PC
over SFTP.

Started once by main_refactored.py at rig startup and left running for
the lifetime of the rig process: it connects, watches the images folder,
and transfers anything that appears there.

Credentials are read from CONFIG_PATH (see prompt_and_save_credentials
for the first-run flow). That file holds a password/passphrase in
plaintext, so it is created with 0600 permissions and the script warns
if it is found more permissive than that.
"""

from __future__ import annotations

import configparser
import logging
import os
import stat
import sys
import time
from getpass import getpass
from pathlib import Path, PureWindowsPath

import paramiko
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from alignment_common import EXIT_CONNECTION_FAILED, get_logger

logger = get_logger(__name__, level=logging.INFO)

CONFIG_PATH = Path.home() / ".config" / "dfaa" / "pc_connection.ini"
CONFIG_SECTION = "pc"

WATCH_DIR = Path.home() / "Documents" / "images"  # on the Raspberry Pi
DESTINATION_DIR = PureWindowsPath(r"C:\Users\joaos\Desktop\Images")  # on the Windows PC

CONNECT_TIMEOUT_S = 10
FILE_SETTLE_DELAY_S = 1  # wait for a newly-created file to finish being written before reading it
TRANSFER_RETRIES = 3
TRANSFER_RETRY_DELAY_S = 2


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def prompt_and_save_credentials() -> dict:
    """
    Ask for the PC's SSH details once and write them to CONFIG_PATH.

    Only used when the config file doesn't exist yet. Password and key
    passphrase use getpass() so they aren't echoed to the terminal.
    Leave any field blank if it doesn't apply (e.g. no password when
    authenticating with a key, or no passphrase on an unencrypted key).
    """
    print(f"Configuracao inicial da ligacao ao PC (guardada em {CONFIG_PATH}):")
    credentials = {
        "hostname": input("  Endereco IP / hostname do PC: ").strip(),
        "username": input("  Utilizador: ").strip(),
        "key_filename": input("  Caminho da chave SSH (Enter para nenhum): ").strip(),
        "password": getpass("  Password (Enter para nenhuma): "),
        "passphrase": getpass("  Passphrase da chave (Enter para nenhuma): "),
    }

    config = configparser.ConfigParser()
    config[CONFIG_SECTION] = credentials

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 *before* writing, so the secrets are never briefly
    # readable by other users between creation and a later chmod.
    fd = os.open(CONFIG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        config.write(f)

    logger.info("Credentials written to %s", CONFIG_PATH)
    print(f"Credenciais guardadas. Para as alterar, edite ou apague {CONFIG_PATH}.")
    return _normalize(credentials)


def load_credentials() -> dict:
    """Read credentials from CONFIG_PATH, prompting and creating it on first run."""
    if not CONFIG_PATH.exists():
        logger.info("No credentials file at %s; prompting for initial setup.", CONFIG_PATH)
        return prompt_and_save_credentials()

    mode = CONFIG_PATH.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        # Readable by group/others: the password and passphrase in here are
        # plaintext, so this is worth surfacing rather than silently accepting.
        logger.warning(
            "%s is readable by other users (mode %o). Run: chmod 600 %s",
            CONFIG_PATH, mode & 0o777, CONFIG_PATH,
        )
        print(f"Aviso: {CONFIG_PATH} tem permissoes demasiado abertas. Corra: chmod 600 {CONFIG_PATH}")

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if CONFIG_SECTION not in config:
        logger.error("Config file %s has no [%s] section.", CONFIG_PATH, CONFIG_SECTION)
        print(f"Ficheiro de configuracao invalido: {CONFIG_PATH}. Apague-o para reconfigurar.")
        sys.exit(EXIT_CONNECTION_FAILED)

    section = config[CONFIG_SECTION]
    return _normalize({key: section.get(key, "") for key in
                       ("hostname", "username", "key_filename", "password", "passphrase")})


def _normalize(credentials: dict) -> dict:
    """Blank optional fields become None, which is what paramiko expects for 'not set'."""
    return {
        "hostname": credentials["hostname"],
        "username": credentials["username"],
        "key_filename": credentials["key_filename"] or None,
        "password": credentials["password"] or None,
        "passphrase": credentials["passphrase"] or None,
    }


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

def connect(credentials: dict) -> paramiko.SSHClient:
    """Open the SSH connection, or exit with EXIT_CONNECTION_FAILED."""
    client = paramiko.SSHClient()

    # Only trust hosts already present in ~/.ssh/known_hosts. AutoAddPolicy
    # is deliberately NOT used: it would accept any host key presented,
    # which defeats the point of verifying we're talking to the intended PC.
    #
    # On a fresh Pi image, known_hosts will not yet contain this PC and the
    # connection below will fail. To pre-populate it, connect by hand once
    # from the Pi and accept the fingerprint:
    #     ssh <username>@<hostname>
    # (or: ssh-keyscan <hostname> >> ~/.ssh/known_hosts, if you have already
    # verified the fingerprint by another means).
    client.load_system_host_keys()

    try:
        client.connect(
            credentials["hostname"],
            username=credentials["username"],
            password=credentials["password"],
            key_filename=credentials["key_filename"],
            passphrase=credentials["passphrase"],
            timeout=CONNECT_TIMEOUT_S,         # TCP connection timeout
            banner_timeout=CONNECT_TIMEOUT_S,  # wait for SSH banner
            auth_timeout=CONNECT_TIMEOUT_S,    # authentication timeout
        )
    except Exception as e:
        # Log the exception type/message but never the credentials themselves.
        logger.error("Ligacao SSH com o PC falhou: %s", e)
        print("Ligacao SSH com o PC falhou. Ver main.log para detalhes.")
        sys.exit(EXIT_CONNECTION_FAILED)

    logger.info("A conexao ao PC foi feita com sucesso.")
    return client


# --------------------------------------------------------------------------
# Folder watching / transfer
# --------------------------------------------------------------------------

class FileTransferHandler(FileSystemEventHandler):
    """
    Transfers each newly-created file to the Windows PC.

    FileSystemEventHandler is a base class defining methods called when
    filesystem events occur; it does nothing by itself, so we override
    only the event we care about (file creation).
    """

    def __init__(self, sftp: paramiko.SFTPClient):
        super().__init__()
        self.sftp = sftp

    def on_created(self, event):
        if event.is_directory:
            return

        # Wait a moment so the file is fully written before we read it --
        # on_created fires as soon as the file appears, which can be
        # before the writer has finished.
        time.sleep(FILE_SETTLE_DELAY_S)

        source = Path(event.src_path)
        destination = DESTINATION_DIR / source.name
        logger.info("New file detected: %s", source.name)

        # watchdog calls this on its own background thread. An uncaught
        # exception here would kill that thread silently: the main loop
        # would keep running and the script would look healthy while
        # quietly transferring nothing ever again. So every failure is
        # caught and logged per-file, and a failed file is skipped rather
        # than taking the watcher down with it.
        for attempt in range(1, TRANSFER_RETRIES + 1):
            try:
                self.sftp.put(str(source), str(destination))
                logger.info("File saved in Windows: %s", destination)
                return
            except Exception as e:
                logger.warning(
                    "Transfer of %s failed (attempt %d/%d): %s",
                    source.name, attempt, TRANSFER_RETRIES, e,
                )
                if attempt < TRANSFER_RETRIES:
                    time.sleep(TRANSFER_RETRY_DELAY_S)

        logger.error("Giving up on transferring %s after %d attempts.", source.name, TRANSFER_RETRIES)


def main() -> None:
    credentials = load_credentials()
    client = connect(credentials)

    # SFTP (SSH File Transfer Protocol) session, for transferring files
    # to the remote machine over the existing SSH connection.
    sftp = client.open_sftp()

    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    # An Observer watches a directory for changes but doesn't know what to
    # do about them; the handler supplies that. Wiring them together:
    #
    #   A file is saved
    #           |
    #           v
    #   Observer notices the change
    #           |
    #           v
    #   Calls FileTransferHandler.on_created()
    #           |
    #           v
    #   The file is transferred to the PC
    observer = Observer()
    observer.schedule(FileTransferHandler(sftp), path=str(WATCH_DIR), recursive=True)
    observer.start()

    logger.info("Watching folder: %s", WATCH_DIR)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.debug("Stopping watchdog...")
    finally:
        observer.stop()
        observer.join()
        sftp.close()
        client.close()
        logger.info("SSH connection closed and watchdog stopped.")


if __name__ == "__main__":
    main()
