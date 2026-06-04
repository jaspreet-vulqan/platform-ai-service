"""Orchestrates column type/date inference.

Flow: build prompt from the input columns -> ask the LLM for per-column
inferences (JSON) -> parse + validate -> merge back onto the original columns ->
return a Result envelope.

``parseInference`` and ``buildOutput`` are pure (no vLLM) so they can be unit
tested without a GPU; the LLM call is imported lazily inside ``inferSchema``.
"""

from collections import defaultdict, deque

from ai.template.SchemaInferenceTemplate import (
    SchemaInferenceSystemPrompt,
    buildUserPrompt,
)
from helper.JSONHelper import extractJSON
from helper.LoggingHelper import getLogger
from model.ResultModel import Result
from model.SchemaModel import (
    FileSchemaInput,
    FileSchemaOutput,
    InferenceResult,
    InferredColumn,
)

logger = getLogger(__name__)

_DATE_TYPES = {"date", "datetime"}


def parseInference(text: str) -> InferenceResult:
    """Parse + validate the LLM's JSON into InferenceResult. Raises on failure."""
    return InferenceResult.model_validate(extractJSON(text))


def buildOutput(
    file_schema: FileSchemaInput, inference: InferenceResult
) -> FileSchemaOutput:
    """Merge inferences onto the original columns, preserving order and originals.

    Matches by ColumnName (queue per name handles duplicate names + reordering).
    Missing columns fall back to string/non-date. isDate and DateFormat are
    re-derived from InferredDataType so they are always internally consistent.
    """
    queues: dict[str, deque] = defaultdict(deque)
    for inf in inference.Columns:
        queues[inf.ColumnName].append(inf)

    out_columns = []
    for col in file_schema.Columns:
        queue = queues.get(col.ColumnName)
        inf = queue.popleft() if queue else None

        if inf is not None:
            inferred_type = inf.InferredDataType
        else:
            logger.warning("No inference returned for column '%s'", col.ColumnName)
            inferred_type = "string"

        is_date = inferred_type in _DATE_TYPES
        date_format = (inf.DateFormat if (inf and is_date) else "")

        out_columns.append(
            InferredColumn(
                ColumnName=col.ColumnName,
                DataType=col.DataType,
                SampleValues=col.SampleValues,
                InferredDataType=inferred_type,
                isDate=is_date,
                DateFormat=date_format,
            )
        )

    return FileSchemaOutput(FileName=file_schema.FileName, Columns=out_columns)


def _maxTokensFor(num_columns: int) -> int:
    # Each column's JSON inference is small (~40 tokens); budget generously.
    return min(4096, 256 + 80 * max(1, num_columns))


async def inferSchema(file_schema: FileSchemaInput) -> Result:
    # Lazy import keeps the HTTP/backend deps out of this module's import path,
    # so parseInference/buildOutput stay unit-testable without a running backend.
    from engine.LLMClient import chatComplete

    try:
        columns = [c.model_dump() for c in file_schema.Columns]
        messages = [
            {"role": "system", "content": SchemaInferenceSystemPrompt},
            {"role": "user", "content": buildUserPrompt(columns)},
        ]
        max_tokens = _maxTokensFor(len(file_schema.Columns))

        result = await chatComplete(
            messages, temperature=0.0, max_tokens=max_tokens
        )
        if result.Status != 1:
            return Result(Status=0, Message=result.Message)

        try:
            inference = parseInference(result.Data)
        except Exception as parse_err:
            # One corrective retry — local models occasionally add stray prose.
            logger.warning("First inference parse failed (%s); retrying.", parse_err)
            messages.append({"role": "assistant", "content": result.Data or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not valid. Respond with ONLY the JSON object, "
                        "exactly one entry per column, no markdown or commentary."
                    ),
                }
            )
            result = await chatComplete(
                messages, temperature=0.0, max_tokens=max_tokens
            )
            if result.Status != 1:
                return Result(Status=0, Message=result.Message)
            inference = parseInference(result.Data)

        output = buildOutput(file_schema, inference)
        return Result(
            Data=output, Status=1, Message="JSON updated successfully!"
        )
    except Exception as ex:
        logger.exception("Schema inference failed")
        return Result(Status=0, Message=f"Error inferring schema: {ex}")
