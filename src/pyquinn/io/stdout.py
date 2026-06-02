"""Handles output to stdout."""

import sys
from collections.abc import Iterable


def progress_bar(
    iterable: Iterable,
    total: int | None = None,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 0,
    length: int = 80,
    fill: str = "█",
    print_end: str = "\r",
):
    """Prints a progress bar to stdout while iterating over an iterable object.

    Args:
        iterable (Iterable): An iterable object.
        total (int, optional): Length of iterable object. If None, the len()
            function is used. Defaults to None.
        prefix (str, optional): Prefix string. Defaults to ''.
        suffix (str, optional): Suffix string. Defaults to ''.
        decimals (int, optional): Number of decimal places. Defaults to 1.
        length (int, optional): Character length of bar. Defaults to 100.
        fill (str, optional): Bar fill character. Defaults to '█'.
        print_end (str, optional): End character (e.g. "\r", "\r\n"). Defaults
            to "\r".

    Yields:
        object: items from iterable.
    """

    if total is None:
        total = len(iterable)

    def print_progress(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print(
            f"\r{prefix} {percent}%|{bar}| {iteration}/{total} {suffix}",
            end=print_end,
        )

    print_progress(0)

    for i, item in enumerate(iterable):
        yield item
        print_progress(i + 1)

    print()


def print_overwrite(message: str, *, flush: bool = True) -> None:
    """
    Prints `message` to stdout and keeps it on one line by returning
    the cursor to the beginning. Useful for progress percentages, losses, etc.
    """
    print(f"\r{message}", end="")
    if flush:
        sys.stdout.flush()
