"""Phase 3b: /global route plumbs the model artifact through and renders
a stub-banner when LightGBM pickles are absent."""
from __future__ import annotations

import pytest

from india_quant.dashboard.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_stub_banner_rendered_when_pickles_missing(client, tmp_path, monkeypatch):
    """With no models/global_tab/*.pkl on disk, app falls back to StubArtifact
    and the route renders the artifact banner advertising "StubArtifact"."""
    # Point cwd at a temp dir so models/global_tab is missing.
    monkeypatch.chdir(tmp_path)

    app = create_app()
    app.config.update(TESTING=True)
    c = app.test_client()
    body = c.get("/global?capital=100000&mode=balanced").get_data(as_text=True)
    assert "StubArtifact" in body


def test_artifact_config_set_at_init():
    app = create_app()
    artifact = app.config.get("GLOBAL_TAB_ARTIFACT")
    assert artifact is not None
    assert getattr(artifact, "name", None) in {"stub", "lightgbm"}
