"""pytest 共通設定 — リポジトリルートを import path に追加."""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ からプロジェクトルートを指すようにして `from src import ...` を有効化
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# RuntimeWarning の抑制は pytest.ini の filterwarnings で行う
# （pytest が conftest.py より先に警告キャプチャを構成するため）
