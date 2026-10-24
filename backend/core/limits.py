from starlette.exceptions import HTTPException


class BodyLimitMiddleware:
    """Bound streamed/chunked request bodies before multipart spooling."""

    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                raise HTTPException(status_code=413, detail="upload_too_large")
            return message

        await self.app(scope, bounded_receive, send)
