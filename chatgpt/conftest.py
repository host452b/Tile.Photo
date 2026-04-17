import sys
from pathlib import Path

# Add the project root to sys.path so tests can import mosaic_core
sys.path.insert(0, str(Path(__file__).parent))
