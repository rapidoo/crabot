"""Tests for loop detection."""

from agent.intelligence.loop_detection import LoopDetector


class TestLoopDetector:
    def test_no_loop(self):
        ld = LoopDetector()
        assert ld.check("code", "print(1)") is None
        assert ld.check("file", "read:x") is None
        assert ld.check("search", "query") is None

    def test_repeat_detected(self):
        ld = LoopDetector(max_repeat=3)
        assert ld.check("code", "same") is None
        assert ld.check("code", "same") is None
        result = ld.check("code", "same")
        assert result is not None
        assert "Repeat" in result

    def test_repeat_different_input_ok(self):
        ld = LoopDetector(max_repeat=3)
        assert ld.check("code", "input1") is None
        assert ld.check("code", "input2") is None
        assert ld.check("code", "input3") is None  # different inputs, no loop

    def test_circuit_breaker(self):
        ld = LoopDetector(circuit_breaker=5)
        for i in range(5):
            assert ld.check("code", f"input{i}") is None
        result = ld.check("code", "one_more")
        assert result is not None
        assert "Circuit breaker" in result

    def test_ping_pong(self):
        ld = LoopDetector(ping_pong_depth=2)
        assert ld.check("code", "a") is None
        assert ld.check("file", "b") is None
        assert ld.check("code", "c") is None
        result = ld.check("file", "d")
        assert result is not None
        assert "Ping-pong" in result

    def test_reset(self):
        ld = LoopDetector(max_repeat=2)
        ld.check("code", "same")
        ld.check("code", "same")  # would trigger on next
        ld.reset()
        assert ld.check("code", "same") is None  # fresh start

    def test_no_false_positive_mixed(self):
        ld = LoopDetector()
        for i in range(15):
            tool = ["code", "file", "search"][i % 3]
            assert ld.check(tool, f"input{i}") is None
