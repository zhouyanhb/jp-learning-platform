"""Command line entrypoint for JP Learning Platform."""

from __future__ import annotations

import logging

from jp_learning_platform import __version__

LOGGER = logging.getLogger(__name__)


def main() -> int:
    LOGGER.info("jp-learning-platform %s", __version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
