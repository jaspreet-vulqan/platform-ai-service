"""Real API-key authentication.

The sample's ``validateHeaders`` accepted every request (always Status=1). This
replaces it with an actual check usable as a FastAPI dependency.

Accepted credential headers (either works):
    Authorization: Bearer <key>
    X-API-Key: <key>

If ``settings.API_KEYS`` is empty, auth is disabled (intended for local dev
only — set keys in production).
"""

from fastapi import Header, HTTPException, status

import settings


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return authorization.strip()
    if x_api_key:
        return x_api_key.strip()
    return None


async def requireApiKey(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency: 401 unless a configured key is presented."""
    if not settings.API_KEYS:
        return  # auth disabled

    key = _extract_key(authorization, x_api_key)
    if key is None or key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
