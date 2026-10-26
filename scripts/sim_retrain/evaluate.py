"""Delegate to the unchanged reproduction entry point."""
import runpy,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"reproduce"
sys.path.insert(0,str(p))
runpy.run_path(str(p/"evaluate.py"),run_name="__main__")
