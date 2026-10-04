import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'models' are resolved on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI instance
from app import app as fastapi_app


class VercelPathNormalizer:
    """
    ASGI middleware that restores original request paths preserved by Vercel
    in x-matched-path or x-forwarded-uri headers.
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            matched = (
                headers.get(b"x-matched-path", b"")
                or headers.get(b"x-forwarded-uri", b"")
            ).decode("latin1")

            if matched and not matched.endswith(".py"):
                clean = matched.split("?")[0]
                scope["path"] = clean
                scope["raw_path"] = clean.encode("latin1")
        await self.asgi_app(scope, receive, send)


app = VercelPathNormalizer(fastapi_app)
