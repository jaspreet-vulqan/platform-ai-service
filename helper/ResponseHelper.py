"""Builders for the ``Result`` envelope, used by auxiliary (non-OpenAI) endpoints.

The inference endpoints deliberately do NOT use this — they return native
OpenAI-compatible shapes so consumers can use the stock OpenAI SDK.
"""

from typing import Optional

from model.ResultModel import Result


def ok(data=None, message: str = "Operation completed successfully!") -> Result:
    return Result(Data=data, Status=1, Message=message)


def fail(message: str, data=None) -> Result:
    return Result(Data=data, Status=0, Message=message)


def openai_error(message: str, code: str = "internal_error", type_: str = "BadRequest") -> dict:
    """Shape an error like the OpenAI API does, for the /v1/* endpoints."""
    return {
        "error": {
            "message": message,
            "type": type_,
            "code": code,
            "param": None,
        }
    }
