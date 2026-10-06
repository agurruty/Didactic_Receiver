#!/usr/bin/env python3
"""Navigation milestone. Run with --help for parameters; --no-show saves figures only."""
from receiver.common import launch


def main():
    from receiver.navigation import main as run

    return run()


if __name__ == "__main__":
    launch(main)
