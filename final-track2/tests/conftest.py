"""Import the student interface when tests are launched from either repo root."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/student'),str(ROOT)]
