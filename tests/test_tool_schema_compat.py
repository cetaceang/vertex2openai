import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tool_schema_compat import rewrite_vertex_tool_schemas


class ToolSchemaCompatTests(unittest.TestCase):
    def test_untyped_and_union_properties_remain_unconstrained_and_union_schemas(self):
        body = json.dumps({"tools": [{"functionDeclarations": [{"name": "cron", "parameters": {
            "type": "OBJECT", "properties": {
                "sessionKey": {},
                "target": {"anyOf": [{"type": "STRING"}, {"type": "NULL"}]},
                "job": {"properties": {"enabled": {"type": "BOOLEAN"}}},
            }, "required": ["target"],
        }}]}], "contents": [{"parts": [{"text": "unchanged"}]}]}).encode()
        rewritten, changed = rewrite_vertex_tool_schemas(body)
        payload = json.loads(rewritten)
        declaration = payload["tools"][0]["functionDeclarations"][0]
        self.assertEqual(changed, 1)
        self.assertNotIn("parameters", declaration)
        parameters = declaration["parametersJsonSchema"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["properties"]["sessionKey"], {})
        self.assertEqual(parameters["properties"]["target"], {"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(parameters["properties"]["job"], {"properties": {"enabled": {"type": "boolean"}}})
        self.assertEqual(parameters["required"], ["target"])
        self.assertEqual(payload["contents"], json.loads(body)["contents"])

    def test_nullable_enum_keeps_null_valid_and_literal_values_are_untouched(self):
        literal = {"type": "STRING", "nullable": True, "properties": {"type": "literal"}}
        body = json.dumps({"tools": [{"function_declarations": [{"name": "test", "parameters": {
            "type": "OBJECT", "properties": {
                "value": {"type": "STRING", "enum": ["a"], "nullable": True},
                "payload": {"type": "OBJECT", "default": literal},
            },
        }}]}]}).encode()
        rewritten, changed = rewrite_vertex_tool_schemas(body)
        self.assertEqual(changed, 1)
        properties = json.loads(rewritten)["tools"][0]["function_declarations"][0]["parametersJsonSchema"]["properties"]
        self.assertEqual(properties["value"], {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "null"}]})
        self.assertEqual(properties["payload"]["default"], literal)

    def test_existing_json_schema_and_non_tool_bodies_are_byte_preserved(self):
        bodies = [
            b'not json', b'[]', b'{ "contents": [] }\n', b'{ "tools": [{"googleSearch": {}}] }',
            b'{ "tools": [{"functionDeclarations":[{"parametersJsonSchema":{"properties":{"x":{}}}}]}] }',
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.assertEqual(rewrite_vertex_tool_schemas(body), (body, 0))

    def test_vertex_string_limits_become_json_schema_integers(self):
        body = json.dumps({"tools": [{"functionDeclarations": [{"name": "test", "parameters": {
            "type": "OBJECT", "minProperties": "1", "properties": {
                "values": {"type": "ARRAY", "minItems": "1", "maxItems": "3", "items": {
                    "type": "STRING", "minLength": "2", "maxLength": "10",
                }},
            },
        }}]}]}).encode()
        rewritten, _ = rewrite_vertex_tool_schemas(body)
        schema = json.loads(rewritten)["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
        self.assertEqual(schema["minProperties"], 1)
        values = schema["properties"]["values"]
        self.assertEqual((values["minItems"], values["maxItems"]), (1, 3))
        self.assertEqual(values["items"], {"type": "string", "minLength": 2, "maxLength": 10})


if __name__ == "__main__":
    unittest.main()
