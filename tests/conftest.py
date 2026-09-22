"""Pytest configuration: make ``notebooks/utils`` importable as ``metrics``."""
import os
import sys

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "utils")
sys.path.insert(0, os.path.abspath(_UTILS_DIR))
