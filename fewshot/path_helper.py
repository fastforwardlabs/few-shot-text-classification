"""Gives a platform-independent way to get files."""
import os
import pathlib


def check_path(pathname: str) -> bool:
    newdir = "/".join(pathname.split("/")[:-1])
    if not os.path.exists(newdir):
        os.makedirs(newdir)


def fewshot_filename(*paths) -> str:
    """Given a path relative to this project's top-level directory, returns the
    full path in the OS.

    Args:
        paths: A list of folders/files.  These will be joined in order with "/"
            or "\" depending on platform.

    Returns:
        The full absolute path in the OS.
    """
    # First parent gets the scripts directory, and the second gets the top-level.
    result_path = pathlib.Path(__file__).resolve().parent.parent
    for path in paths:
        result_path /= path
    return str(result_path)
