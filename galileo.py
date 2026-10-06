#!/usr/bin/env python3
"""Galileo E1/E5 teaching receiver. Use --help; --no-show saves plots headlessly."""

from receiver.common import launch

if __name__ == "__main__":
    from receiver.multiband_cli import main

    launch(main)
