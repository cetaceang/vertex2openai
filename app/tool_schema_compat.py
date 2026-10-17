"""Opt-in migration from Vertex Schema to JSON Schema for native tool inputs."""

import json


def _json_schema(schema):
    if not isinstance(schema, dict):
        return schema
    result = schema.copy()
    schema_type = result.get("type")
    if isinstance(schema_type, str):
        if schema_type.upper() == "TYPE_UNSPECIFIED":
            result.pop("type")
        elif schema_type.upper() in {"STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT", "NULL"}:
            result["type"] = schema_type.lower()
    # Vertex Schema encodes its int64 limits as JSON strings; JSON Schema
    # requires integer values for these same keywords.
    for key in ("minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"):
        if isinstance(result.get(key), str):
            try:
                result[key] = int(result[key])
            except ValueError:
                pass

    # Recurse only into schema positions. A default/enum/example may itself
    # contain objects named 'type', 'properties', or 'nullable'.
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        if isinstance(result.get(key), dict):
            result[key] = {name: _json_schema(value) for name, value in result[key].items()}
    for key in ("items", "additionalProperties", "not", "contains", "if", "then", "else", "propertyNames"):
        if key in result:
            result[key] = ([_json_schema(value) for value in result[key]]
                           if isinstance(result[key], list) else _json_schema(result[key]))
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(result.get(key), list):
            result[key] = [_json_schema(value) for value in result[key]]

    nullable = result.pop("nullable", False)
    if nullable is True:
        # anyOf preserves existing enum/const/composition constraints and adds
        # null without guessing the type of an unconstrained schema such as {}.
        result = {"anyOf": [result, {"type": "null"}]}
    return result


def rewrite_vertex_tool_schemas(body):
    """Return the original bytes when there is nothing to migrate.

    parametersJsonSchema accepts JSON Schema's unconstrained nodes and unions;
    adding an arbitrary 'type: string' to them would change tool semantics.
    The caller opts into this transformation; strict passthrough is the default.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return body, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        return body, 0
    changed = 0
    for tool in payload["tools"]:
        if not isinstance(tool, dict):
            continue
        declarations = tool.get("functionDeclarations", tool.get("function_declarations"))
        if not isinstance(declarations, list):
            continue
        for declaration in declarations:
            if not isinstance(declaration, dict) or not isinstance(declaration.get("parameters"), dict):
                continue
            if "parametersJsonSchema" in declaration or "parameters_json_schema" in declaration:
                continue
            declaration["parametersJsonSchema"] = _json_schema(declaration.pop("parameters"))
            changed += 1
    if not changed:
        return body, 0
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), changed
