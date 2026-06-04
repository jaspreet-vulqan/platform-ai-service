"""Column type/date inference endpoint.

POST /api/v1/infer-schema
    Body : FileSchemaInput  (FileName + Columns[ColumnName, DataType, SampleValues])
    Reply: Result[FileSchemaOutput] -> {"Data": <augmented schema>, "Status": 1,
           "Message": "JSON updated successfully!"}

Returns our custom Result envelope (not an OpenAI shape) — it's an internal
feature, not part of the proxied OpenAI surface.
"""

from fastapi import APIRouter, Depends

from engine.SchemaInferenceClient import inferSchema
from model.ResultModel import Result
from model.SchemaModel import FileSchemaInput, FileSchemaOutput
from server.ValidateRequest import requireApiKey

router = APIRouter()


@router.post(
    "/api/v1/infer-schema",
    response_model=Result[FileSchemaOutput],
    dependencies=[Depends(requireApiKey)],
)
async def infer_schema(payload: FileSchemaInput) -> Result:
    return await inferSchema(payload)
