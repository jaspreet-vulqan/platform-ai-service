"""Prompt templates for column data-type / date inference.

Kept in-repo under ai/template/ (same convention as the sample app's PGTemplate
etc.) so prompts are versioned alongside the code. When this feature moves to a
separate service, these templates move with it.

Date formats use Unicode/LDML (Java/.NET DateTimeFormatter) tokens.
"""

import json
from typing import List

SchemaInferenceSystemPrompt = (
    "You are a precise data type inference engine. You examine the sample values "
    "of each column in a tabular file and determine the true underlying data "
    "type, paying special attention to dates and timestamps. You always respond "
    "with strict, valid JSON only - no prose, no markdown."
)

# Static instruction block. NOTE: contains literal JSON braces, so it is built by
# concatenation (not str.format) to avoid brace-escaping headaches.
_INSTRUCTIONS = """You are given the columns of a file. Each column has a ColumnName, a DataType reported by the source system (often just "string"), and a few SampleValues.

For EACH column, look ONLY at the SampleValues and infer the true data type. Return one result object per column with these fields:
- "ColumnName": echo the input ColumnName exactly.
- "InferredDataType": one of exactly ["string","integer","float","boolean","date","datetime"].
    - "integer"  : whole numbers, even when stored as strings (e.g. "42", "1001").
    - "float"    : decimal numbers, even when stored as strings (e.g. "3.14", "10.0").
    - "boolean"  : true/false, yes/no, or 0/1 used as flags.
    - "date"     : a calendar date with NO time component (e.g. "2025/09/13").
    - "datetime" : a date that also includes a time (and optionally a timezone), e.g. "1981-03-01 19:00:00 GMT-0500 (Eastern Standard Time)".
    - "string"   : anything else - names, identifiers containing letters, emails, free text.
- "isDate": true if InferredDataType is "date" or "datetime", otherwise false.
- "DateFormat": when isDate is true, the format pattern of the sample values written with Unicode/LDML (Java/.NET DateTimeFormatter) tokens:
      yyyy = 4-digit year, MM = 2-digit month, dd = 2-digit day,
      HH = 24-hour hour, mm = minute, ss = second,
      OOOO = localized GMT offset (e.g. GMT-05:00), zzzz = full time-zone name.
  The pattern MUST cover the ENTIRE sample string, including any time and time-zone parts.
  When isDate is false, DateFormat MUST be "".

Rules:
- Decide from the SampleValues, NOT from the ColumnName or the source DataType.
- A value like "Cus 103" or an email address is a string, not a number or date.
- If the sample values disagree, pick the type that fits all of them; fall back to "string" if they cannot agree.
- Output EXACTLY one object per input column, in the same order.
- Respond with ONLY the JSON object below - no markdown fences, no commentary.

Output shape:
{"Columns":[{"ColumnName":"...","InferredDataType":"...","isDate":false,"DateFormat":""}]}

Example input columns:
[{"ColumnName":"Customer ID","DataType":"string","SampleValues":["Cus101","Cus102","Cus 103"]},{"ColumnName":"Date of Birth","DataType":"string","SampleValues":["1981-03-01 19:00:00 GMT-0500 (Eastern Standard Time)","2083-01-21 14:20:00 GMT-0500 (Eastern Standard Time)"]},{"ColumnName":"Joining Date","DataType":"string","SampleValues":["2025/09/13","2025/12/21"]}]
Example output:
{"Columns":[{"ColumnName":"Customer ID","InferredDataType":"string","isDate":false,"DateFormat":""},{"ColumnName":"Date of Birth","InferredDataType":"datetime","isDate":true,"DateFormat":"yyyy-MM-dd HH:mm:ss OOOO (zzzz)"},{"ColumnName":"Joining Date","InferredDataType":"date","isDate":true,"DateFormat":"yyyy/MM/dd"}]}"""


def buildUserPrompt(columns: List[dict]) -> str:
    """Render the user prompt for the given columns.

    Args:
        columns: list of dicts with ColumnName / DataType / SampleValues.
    """
    columns_json = json.dumps(columns, ensure_ascii=False)
    return f"{_INSTRUCTIONS}\n\nNow infer for these columns:\n{columns_json}"
