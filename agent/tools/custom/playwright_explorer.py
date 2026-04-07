"""Auto-generated tool: playwright_explorer — Un outil de navigation web autonome utilisant Playwright pour explorer, cliquer, remplir des formulaires et extraire du contenu web."""

from __future__ import annotations

from agent.tools.registry import register_tool


class PlaywrightExplorerTool:
    """Tool: Un outil de navigation web autonome utilisant Playwright pour explorer, cliquer, remplir des formulaires et extraire du contenu web."""

    @property
    def name(self) -> str:
        return "playwright_explorer"

    @property
    def description(self) -> str:
        return "Un outil de navigation web autonome utilisant Playwright pour explorer, cliquer, remplir des formulaires et extraire du contenu web."

    async def run(self, input: str) -> str:
        import asyncio
        from playwright.async_api import async_playwright
        
        async def run(params):
            url = params.get('url')
            action = params.get('action', 'goto')
            selector = params.get('selector')
            text = params.get('text')
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                if url:
                    await page.goto(url)
                
                if action == 'click' and selector:
                    await page.click(selector)
                elif action == 'fill' and selector and text:
                    await page.fill(selector, text)
                elif action == 'screenshot':
                    await page.screenshot(path='screenshot.png')
                    return 'Capture d\' enregistr sous screenshot.png'
                
                content = await page.content()
                await browser.close()
                return content[:10000] # Retourne les 10000 premiers caract


def _factory() -> PlaywrightExplorerTool:
    return PlaywrightExplorerTool()


register_tool("playwright_explorer", _factory)
