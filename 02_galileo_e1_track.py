#!/usr/bin/env python3
"""Tracking milestone. Run with --help for parameters; --no-show saves figures only."""
from receiver.common import launch


def main():
    from receiver.tracking import main as run

    return run()


if __name__ == "__main__":
    launch(main)
