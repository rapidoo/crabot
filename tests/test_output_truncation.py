"""Tests for output truncation."""

from agent.core.output_truncation import truncate_output


class TestOutputTruncation:
    def test_short_output_unchanged(self):
        output = "hello world"
        result = truncate_output(output, max_chars=100)
        assert result == output

    def test_exact_limit_unchanged(self):
        output = "x" * 100
        result = truncate_output(output, max_chars=100)
        assert result == output

    def test_long_output_truncated(self):
        output = "A" * 1000 + "B" * 1000
        result = truncate_output(output, max_chars=500, tail_chars=100)

        assert len(result) < len(output)
        assert "truncated" in result
        # Head should be preserved
        assert result.startswith("A" * 100)
        # Tail should be preserved
        assert result.endswith("B" * 100)

    def test_truncation_indicator_shows_count(self):
        output = "x" * 10_000
        result = truncate_output(output, max_chars=1000, tail_chars=200)
        # Should show how many chars were cut
        assert "9,000" in result  # 10000 - 1000 = 9000 cut

    def test_default_limits(self):
        # Under 50K should pass through
        output = "y" * 49_999
        assert truncate_output(output) == output

        # Over 50K should truncate
        output = "z" * 60_000
        result = truncate_output(output)
        assert "truncated" in result

    def test_preserves_head_and_tail(self):
        head = "HEAD_CONTENT_" * 100
        middle = "MIDDLE_" * 10000
        tail = "TAIL_CONTENT_" * 100
        output = head + middle + tail

        result = truncate_output(output, max_chars=5000, tail_chars=1000)
        assert "HEAD_CONTENT_" in result
        assert "TAIL_CONTENT_" in result
