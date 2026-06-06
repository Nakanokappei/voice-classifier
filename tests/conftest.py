"""pytest shared setup — adds the project root to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the project root so `from src import ...` resolves inside tests.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# RuntimeWarning suppression lives in pytest.ini via `filterwarnings`,
# because pytest wires up its warnings capture before this conftest runs.


@pytest.fixture(autouse=True)
def _stub_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide harmless Azure OpenAI defaults so tests don't need to set them.

    Tests that mock ``_make_azure_client`` never reach the real client, but
    deployment-name resolution (``namer._resolve_deployment``,
    ``advisor._default_deployment``) is exercised eagerly and would otherwise
    fail with a ``ValueError`` whenever a test omits ``model=``.

    Tests that intentionally exercise the missing-env path (e.g.
    ``test_missing_api_key_raises``) override or delete these via their own
    monkeypatch calls.
    """
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "test-embedding")
    monkeypatch.setenv("AZURE_OPENAI_NAMER_DEPLOYMENT", "test-namer")
    monkeypatch.setenv("AZURE_OPENAI_ADVISOR_DEPLOYMENT", "test-advisor")
