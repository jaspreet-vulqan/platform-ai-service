from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """Uniform response envelope reused from the sample app.

    Used only for our auxiliary endpoints (/info, errors). The OpenAI-compatible
    inference endpoints return native OpenAI shapes, not this envelope.

    NOTE: unlike the sample, this is a real ``Generic[T]`` so ``Data`` is actually
    type-checked/validated against the parameterized type.
    """

    Data: Optional[T] = None
    Status: int
    Message: str
