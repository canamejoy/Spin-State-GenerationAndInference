"""Pytest configuration: make ``common/utils`` importable as ``metrics``."""
import os
import sys

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "utils")
sys.path.insert(0, os.path.abspath(_UTILS_DIR))

# The Monte Carlo engine and its ablation variants live in section 0, beside
# the Modal entry points.
_SIM_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "section0-mcmc.gpu", "codigo")
sys.path.insert(0, os.path.abspath(_SIM_DIR))
