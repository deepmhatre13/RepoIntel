"""Tests for the configuration-loading fix.

Verifies the intended configuration mechanism: the project .env is loaded into
os.environ during application startup so that LLMClient sees GOOGLE_API_KEY
without any manual shell export.

Security: tests never print or assert the real .env secret value. They use a
fake sentinel ("test-google-key") and clear/restore real env vars.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from packages.agentic.llm_client import LLMClient

_FAKE_KEY = "test-google-key"
_FAKE_MODEL = "gemini-1.5-flash"


class TestConfigLoadsDotenv:
    """Verify the centralized config layer loads .env and populates os.environ."""

    def test_dotenv_is_loaded_from_project_env(self, tmp_path, monkeypatch):
        """A .env file at the configured location populates os.environ."""
        from packages.shared import config as app_config

        env_file = tmp_path / ".env"
        env_file.write_text(f"GOOGLE_API_KEY={_FAKE_KEY}\nGEMINI_MODEL=gemini-1.5-pro\n")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with mock.patch.object(app_config, "_ENV_PATH", env_file):
            loaded = app_config.Settings.load_dotenv()
        assert loaded is True
        assert os.getenv("GOOGLE_API_KEY") == _FAKE_KEY

    def test_llm_client_sees_key_after_config_load(self, tmp_path, monkeypatch):
        """LLMClient() resolves GOOGLE_API_KEY through the configuration layer."""
        from packages.shared import config as app_config

        env_file = tmp_path / ".env"
        env_file.write_text(f"GOOGLE_API_KEY={_FAKE_KEY}\nGEMINI_MODEL=gemini-1.5-pro\n")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with mock.patch.object(app_config, "_ENV_PATH", env_file):
            app_config.settings.load_dotenv()
            llm = LLMClient()
        assert llm.is_available()
        assert llm.model_name == "gemini-1.5-pro"

    def test_explicit_api_key_overrides_config(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        llm = LLMClient(api_key=_FAKE_KEY)
        assert llm.is_available()
        assert llm.api_key == _FAKE_KEY

    def test_no_key_means_degraded(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        from packages.shared.config import Settings
        Settings._loaded = False
        with mock.patch("packages.shared.config._ENV_PATH", Path("/nonexistent/.env")):
            llm = LLMClient()
        assert not llm.is_available()
        assert llm.api_key == ""


class TestLlmClientModelConfiguration:
    """Verify GEMINI_MODEL resolution precedence (explicit arg > env > default)."""

    def test_explicit_model_name_wins(self):
        llm = LLMClient(api_key=_FAKE_KEY, model_name="gemini-2.0-flash")
        assert llm.model_name == "gemini-2.0-flash"

    def test_falls_back_to_default_model(self, monkeypatch):
        from packages.shared import config as app_config

        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        with mock.patch.object(app_config, "_ENV_PATH", Path("/nonexistent/.env")):
            app_config.Settings._loaded = False
            llm = LLMClient(api_key=_FAKE_KEY)
        assert llm.model_name == "gemini-3.8-flash"
        assert llm.model_name != "gemini-flash-latest"

    def test_env_var_model_respected(self, monkeypatch):
        from packages.shared import config as app_config

        monkeypatch.setenv("GEMINI_MODEL", _FAKE_MODEL)
        with mock.patch.object(app_config, "_ENV_PATH", Path("/nonexistent/.env")):
            app_config.Settings._loaded = False
            llm = LLMClient(api_key=_FAKE_KEY)
        assert llm.model_name == _FAKE_MODEL


class TestApiLlmStatus:
    """Verify the /llm-status endpoint reflects .env configuration."""

    def test_status_reports_degraded_without_key(self, monkeypatch):
        from fastapi.testclient import TestClient
        from apps.api import app
        from packages.shared import config as app_config

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with mock.patch.object(
            app_config, "_ENV_PATH", Path("/nonexistent/.env")
        ):
            app_config.Settings._loaded = False
            with TestClient(app) as client:
                resp = client.get("/llm-status")
                data = resp.json()
        assert data["mode"] == "DEGRADED_NO_LLM"
        assert data["llm_available"] is False
        assert data["provider"] is None

    def test_status_reports_live_with_key(self, tmp_path):
        from fastapi.testclient import TestClient
        from apps.api import app
        from packages.shared import config as app_config

        env_file = tmp_path / ".env"
        env_file.write_text(f"GOOGLE_API_KEY={_FAKE_KEY}\nGEMINI_MODEL=gemini-1.5-pro\n")
        with mock.patch.object(app_config, "_ENV_PATH", env_file):
            app_config.settings.load_dotenv()
            with TestClient(app) as client:
                resp = client.get("/llm-status")
                data = resp.json()
        assert data["mode"] == "LIVE_LLM"
        assert data["llm_available"] is True
        assert data["provider"] == "gemini"


class TestSubprocessConfigLoading:
    """Verify .env is loaded in a fresh process — proving the fix does not
    depend on the developer shell environment."""

    FAKE_KEY = "test-google-key"

    _REPO_ROOT = str(Path(__file__).resolve().parent.parent)

    def _clean_env(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_MODEL", "REPOINTEL_ENV_FILE")}
        # Make the RepoIntel package importable from the subprocess cwd.
        env["PYTHONPATH"] = self._REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def test_fresh_process_sees_fake_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "GOOGLE_API_KEY=" + self.FAKE_KEY + "\nGEMINI_MODEL=gemini-1.5-pro\n"
            )
            env = self._clean_env()
            # Point the config layer at the temp .env via the documented override.
            env["REPOINTEL_ENV_FILE"] = str(env_file)
            result = subprocess.run(
                [sys.executable, "-c",
                 "from packages.shared.config import settings; settings.load_dotenv();"
                 "from packages.agentic.llm_client import LLMClient;"
                 "llm = LLMClient();"
                 "assert llm.is_available(), 'KEY NOT LOADED';"
                 "assert llm.api_key == 'test-google-key';"
                 "print('SUBPROCESS_OK')"],
                cwd=tmpdir, capture_output=True, text=True,
                env=env, timeout=60,
            )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert "SUBPROCESS_OK" in result.stdout

    def test_fresh_process_degraded_without_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No .env exists in tmpdir; point the config layer there so the
            # real project .env cannot leak into this negative-case test.
            env = self._clean_env()
            env["REPOINTEL_ENV_FILE"] = str(Path(tmpdir) / ".env")
            result = subprocess.run(
                [sys.executable, "-c",
                 "from packages.shared.config import settings;"
                 "from packages.agentic.llm_client import LLMClient;"
                 "settings.load_dotenv();"
                 "llm = LLMClient();"
                 "assert not llm.is_available(), 'SHOULD BE DEGRADED';"
                 "print('SUBPROCESS_DEGRADED')"],
                cwd=tmpdir, capture_output=True, text=True,
                env=env, timeout=60,
            )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert "SUBPROCESS_DEGRADED" in result.stdout


class TestNoNetworkCall:
    """Verify /llm-status never invokes Gemini (configuration check only)."""

    def test_no_gemini_invocation(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from apps.api import app
        from packages.shared import config as app_config

        env_file = tmp_path / ".env"
        env_file.write_text(f"GOOGLE_API_KEY={_FAKE_KEY}\nGEMINI_MODEL=gemini-1.5-pro\n")
        with mock.patch.object(app_config, "_ENV_PATH", env_file):
            app_config.settings.load_dotenv()
            with TestClient(app) as client:
                import packages.agentic.llm_client as llm_mod
                original = llm_mod.LLMClient

                class SpyClient(original):
                    def complete(self, *a, **kw):
                        raise AssertionError(
                            "Gemini complete() must not be called by /llm-status"
                        )

                with mock.patch.object(llm_mod, "LLMClient", SpyClient):
                    resp = client.get("/llm-status")
                    assert resp.status_code == 200

