"""Adaptador HTTPX ASGI para smoke tests síncronos sem usar TestClient legado."""
import asyncio
import httpx


class LocalTestClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, endpoint, **kwargs):
        async def _perform():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://testserver") as client:
                return await client.request(method, endpoint, **kwargs)
        return asyncio.run(_perform())

    def get(self, endpoint, **kwargs):
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint, **kwargs):
        return self.request("POST", endpoint, **kwargs)
