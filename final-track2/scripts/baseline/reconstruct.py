#!/usr/bin/env python3
"""Reconstruction stages in the independent Fire3D environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from baselines.fire3d_recon.pipeline import main
if __name__ == '__main__': main()
