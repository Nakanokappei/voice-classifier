"""pytest shared setup — adds the project root to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root so `from src import ...` resolves inside tests.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# RuntimeWarning suppression lives in pytest.ini via `filterwarnings`,
# because pytest wires up its warnings capture before this conftest runs.
