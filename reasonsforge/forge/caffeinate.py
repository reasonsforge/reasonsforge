"""Prevent macOS idle sleep during long-running operations."""

import atexit
import platform
import subprocess

_process = None


def hold():
    """Start caffeinate to prevent idle sleep. No-op on non-macOS."""
    global _process
    if _process is not None:
        return
    if platform.system() != "Darwin":
        return
    try:
        _process = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(release)
    except FileNotFoundError:
        pass


def release():
    """Stop caffeinate."""
    global _process
    if _process is not None:
        _process.terminate()
        _process.wait()
        _process = None
