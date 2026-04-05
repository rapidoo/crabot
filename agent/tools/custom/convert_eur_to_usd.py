"""Auto-generated tool: convert_eur_to_usd — Converts EUR to USD using a fixed rate of 1.08"""

from __future__ import annotations

from agent.tools.registry import register_tool


class ConvertEurToUsdTool:
    """Tool: Converts EUR to USD using a fixed rate of 1.08"""

    @property
    def name(self) -> str:
        return "convert_eur_to_usd"

    @property
    def description(self) -> str:
        return "Converts EUR to USD using a fixed rate of 1.08"

    async def run(self, input: str) -> str:
        try:
            amount = float(input.strip())
        except ValueError:
            return "ERROR: please provide a numeric amount in EUR"
        return f"{amount * 1.08:.2f} USD"


def _factory() -> ConvertEurToUsdTool:
    return ConvertEurToUsdTool()


register_tool("convert_eur_to_usd", _factory)
