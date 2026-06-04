"""Schemas for the column type/date inference endpoint.

Input  : a file's columns with source-reported DataType + SampleValues.
Output : the same columns augmented with InferredDataType / isDate / DateFormat.

The LLM only produces the per-column *inference* (ColumnInference); the original
ColumnName / DataType / SampleValues are merged back in code so they can never be
altered or dropped by the model.
"""

from typing import Any, List, Literal

from pydantic import BaseModel, Field

# Allowed inferred types. "date" = calendar date only; "datetime" = date + time
# (and optionally timezone). isDate is true for both.
InferredDataType = Literal[
    "string", "integer", "float", "boolean", "date", "datetime"
]


class ColumnInput(BaseModel):
    ColumnName: str
    DataType: str
    SampleValues: List[Any] = Field(default_factory=list)


class FileSchemaInput(BaseModel):
    FileName: str
    Columns: List[ColumnInput]


class ColumnInference(BaseModel):
    """One entry per column, exactly as returned by the LLM."""

    ColumnName: str
    InferredDataType: InferredDataType
    isDate: bool
    DateFormat: str = ""


class InferenceResult(BaseModel):
    """The JSON envelope the LLM is asked to return."""

    Columns: List[ColumnInference]


class InferredColumn(ColumnInput):
    """Final per-column shape: original fields + the three inferred ones."""

    InferredDataType: str
    isDate: bool
    DateFormat: str = ""


class FileSchemaOutput(BaseModel):
    FileName: str
    Columns: List[InferredColumn]
