#!/usr/bin/env python3
"""Pseudorange and position milestone. See --help; --no-show saves plots only."""
from receiver.common import launch


def main():
    from receiver.positioning import main as run
    return run()


if __name__ == '__main__':
    launch(main)
