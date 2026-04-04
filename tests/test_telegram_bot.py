"""Tests for Telegram bot — unit tests with mocks."""

import pytest

from agent.config import Settings
from agent.interfaces.telegram_bot import _split_message


class TestSplitMessage:
    def test_short_message(self):
        chunks = _split_message("hello", max_len=100)
        assert chunks == ["hello"]

    def test_long_message_splits_at_newline(self):
        text = "line1\nline2\nline3\nline4"
        chunks = _split_message(text, max_len=12)
        assert len(chunks) >= 2
        # Reassembled text should match original
        reassembled = "\n".join(chunks)
        assert "line1" in reassembled
        assert "line4" in reassembled

    def test_exact_limit(self):
        text = "a" * 100
        chunks = _split_message(text, max_len=100)
        assert chunks == [text]

    def test_no_good_newline_hard_cuts(self):
        text = "a" * 200
        chunks = _split_message(text, max_len=100)
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100

    def test_empty_message(self):
        assert _split_message("", max_len=100) == [""]


class TestTelegramConfig:
    def test_default_config(self):
        s = Settings()
        assert s.telegram.bot_token == ""
        assert s.telegram.allowed_users == []
        assert s.telegram.max_message_length == 4000

    def test_daemon_config(self):
        s = Settings()
        assert s.daemon.max_concurrent == 3
        assert s.daemon.request_timeout == 300

    def test_custom_config(self):
        s = Settings(
            telegram={"bot_token": "test_token", "allowed_users": [123, 456]},
            daemon={"max_concurrent": 5},
        )
        assert s.telegram.bot_token == "test_token"
        assert 123 in s.telegram.allowed_users
        assert s.daemon.max_concurrent == 5


class TestTelegramBotInit:
    def test_missing_token_raises(self):
        """Bot should fail to init without a token."""
        try:
            from agent.interfaces.telegram_bot import TelegramBot, HAS_TELEGRAM
            if not HAS_TELEGRAM:
                pytest.skip("python-telegram-bot not installed")

            from unittest.mock import AsyncMock
            agent = AsyncMock()
            settings = Settings()

            with pytest.raises(RuntimeError, match="No Telegram bot token"):
                TelegramBot(agent, settings)
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

    def test_allowed_users_filter(self):
        """Test the _is_allowed method."""
        try:
            from agent.interfaces.telegram_bot import TelegramBot, HAS_TELEGRAM
            if not HAS_TELEGRAM:
                pytest.skip("python-telegram-bot not installed")

            from unittest.mock import AsyncMock
            agent = AsyncMock()
            settings = Settings(
                telegram={"bot_token": "fake_token", "allowed_users": [100, 200]},
            )

            bot = TelegramBot(agent, settings)
            assert bot._is_allowed(100) is True
            assert bot._is_allowed(200) is True
            assert bot._is_allowed(999) is False

        except ImportError:
            pytest.skip("python-telegram-bot not installed")

    def test_no_filter_allows_everyone(self):
        """Empty allowed_users means everyone can use the bot."""
        try:
            from agent.interfaces.telegram_bot import TelegramBot, HAS_TELEGRAM
            if not HAS_TELEGRAM:
                pytest.skip("python-telegram-bot not installed")

            from unittest.mock import AsyncMock
            agent = AsyncMock()
            settings = Settings(
                telegram={"bot_token": "fake_token", "allowed_users": []},
            )

            bot = TelegramBot(agent, settings)
            assert bot._is_allowed(999) is True

        except ImportError:
            pytest.skip("python-telegram-bot not installed")
