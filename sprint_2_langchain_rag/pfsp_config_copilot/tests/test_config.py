from __future__ import annotations

import logging

import pytest

from services import config


def test_get_openai_api_key_returns_environment_value(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_local_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    api_key = config.get_openai_api_key()

    assert api_key == "test-key"


def test_get_openai_api_key_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_local_env", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        config.get_openai_api_key()


def test_configure_logging_creates_file_handler_without_duplication(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    logger = logging.getLogger("config_copilot")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    try:
        configured_logger = config.configure_logging()
        configured_logger.info("logging check")
        handler_count = len(configured_logger.handlers)

        configured_again = config.configure_logging()

        assert (tmp_path / "config_copilot.log").exists()
        assert handler_count >= 2
        assert len(configured_again.handlers) == handler_count
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
