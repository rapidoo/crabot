"""Tests for the ApprovalGate — dangerous action approval."""

from unittest.mock import patch

from agent.core.approval import ApprovalGate


class TestApprovalGate:
    def test_disabled_gate_never_needs_approval(self):
        gate = ApprovalGate(interactive=True, enabled=False)
        assert not gate.needs_approval("code", "print('hi')")
        assert not gate.needs_approval("tool_create", "name:foo")

    def test_code_tool_needs_approval(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert gate.needs_approval("code", "print('hi')")

    def test_tool_create_needs_approval(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert gate.needs_approval("tool_create", "name:foo|description:bar")

    def test_file_read_does_not_need_approval(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert not gate.needs_approval("file", "read:/tmp/test.txt")

    def test_file_write_needs_approval(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert gate.needs_approval("file", "write:/tmp/test.txt:content")

    def test_file_delete_needs_approval(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert gate.needs_approval("file", "delete:/tmp/test.txt")

    def test_search_tool_safe(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        assert not gate.needs_approval("search", "*.py")
        assert not gate.needs_approval("web_search", "python tutorial")

    def test_non_interactive_auto_approves(self):
        gate = ApprovalGate(interactive=False, enabled=True)
        assert gate.request_approval("code", "print('hi')") is True

    def test_interactive_approval_yes(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", return_value="y"):
            assert gate.request_approval("code", "print('hi')") is True

    def test_interactive_approval_no(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", return_value="n"):
            assert gate.request_approval("code", "print('hi')") is False

    def test_interactive_approval_empty_is_no(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", return_value=""):
            assert gate.request_approval("code", "print('hi')") is False

    def test_always_caches_pattern(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", return_value="always"):
            gate.request_approval("code", "print('hi')")

        # Now the same pattern should not need approval
        assert not gate.needs_approval("code", "print('hi')")

    def test_oui_accepted(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", return_value="oui"):
            assert gate.request_approval("code", "print('hi')") is True

    def test_eof_returns_false(self):
        gate = ApprovalGate(interactive=True, enabled=True)
        with patch("builtins.input", side_effect=EOFError):
            assert gate.request_approval("code", "print('hi')") is False
