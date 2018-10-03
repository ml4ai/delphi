""" Script to generate the skeleton of a Flask REST API from an OpenAPI 3.0
YAML specification. """

from ruamel.yaml import YAML
from typing import List


def write_models(yml):
    global_schema_list = []
    schemas = yml["components"]["schemas"]
    schema_lines_dict = {}
    module_lines = []
    module_lines.append(
        """\
from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict"""
    )

    def process_properties(
        schema_name, schema, class_lines, class_declaration
    ):
        class_lines.append("\n\n@dataclass")
        class_lines.append(class_declaration)
        class_lines.append(
            '    """'
            + f"{schema.get('description', f'Placeholder docstring for class {schema_name}.')}"
            + ' """'
        )
        properties = schema["properties"]
        required_properties = schema.get("required", [])

        for property in sorted(
            properties, key=lambda x: x not in required_properties
        ):
            property_ref = properties[property].get("$ref", "").split("/")[-1]
            if property_ref != "" and property_ref not in global_schema_list:
                global_schema_list.append(property_ref)
            property_type = properties[property].get("type", property_ref)
            mapping = {
                "string": "str",
                "integer": "int",
                "None": "",
                "array": "List",
                "boolean": "bool",
                "number": "float",
                "object": "object",
            }
            type_annotation = mapping.get(property_type, property_type)
            if type_annotation == "List":
                property_type = properties[property]["items"]["type"]
                type_annotation = (
                    f"List[{mapping.get(property_type, property_type)}]"
                )

            # TODO - Parse dependencies so that inherited properties are
            # duplicated for child classes.

            if property not in required_properties:
                type_annotation = f"Optional[{type_annotation}]"
            if property == "baseType":
                default_value = f'"{schema_name}"'
            else:
                default_value = "None"
            class_lines.append(
                f"    {property}: {type_annotation} = {default_value}"
            )

    def to_class(schema_name, schema):
        class_lines = []
        if schema.get("type") == "object":
            class_declaration = f"class {schema_name}:"
            process_properties(
                schema_name, schema, class_lines, class_declaration
            )
        elif schema.get("allOf") is not None:
            parents = [
                item["$ref"].split("/")[-1]
                for item in schema["allOf"]
                if item.get("$ref")
            ]
            schema = schema["allOf"][1]
            class_declaration = f"class {schema_name}({','.join(parents)}):"
            process_properties(
                schema_name, schema, class_lines, class_declaration
            )
        elif "enum" in schema:
            class_lines.append("\n\n@unique")
            class_lines.append(f"class {schema_name}(Enum):")
            for option in schema["enum"]:
                class_lines.append(f"    {option} = '{option}'")
        schema_lines_dict[schema_name] = class_lines
        if schema_name not in global_schema_list:
            global_schema_list.append(schema_name)

    for schema_name, schema in schemas.items():
        to_class(schema_name, schema)

    for schema_name in global_schema_list:
        module_lines += schema_lines_dict[schema_name]
    with open("models.py", "w") as f:
        f.write("\n".join(module_lines))


def construct_view_lines(url, metadata) -> List[str]:
    lines = []
    parameters = metadata.pop("parameters", None)
    for http_method in metadata:
        modified_path = url.replace("{", "<string:").replace("}", ">")
        lines.append(
            f"\n\n@app.route('{modified_path}', methods=['{http_method.upper()}'])"
        )
        args = ", ".join(
            [
                part[1:-1] + ": str"
                for part in url.split("/")
                if part.startswith("{")
            ]
        )
        operationId = metadata[http_method]["operationId"]
        lines.append(f"def {operationId}({args}):")
        lines.append(
            '    """' + f" {metadata[http_method]['summary']}" + '"""'
        )
        lines.append("    pass")
    return lines


def write_views(yml):
    paths = yml["paths"]

    views_lines = []
    views_lines.append(
        "\n".join(
            [
                "import uuid",
                "import pickle",
                "from datetime import datetime",
                "from typing import Optional, List",
                "from flask import Flask, jsonify, request",
                "from delphi.icm_api.models import *",
                "app = Flask(__name__)",
            ]
        )
    )

    for url, metadata in paths.items():
        views_lines += construct_view_lines(url, metadata)

    views_lines.append(
        """\

if __name__ == "__main__":
    app.run(debug=True)"""
    )

    with open("views.py", "w") as f:
        f.write("\n".join(views_lines))


if __name__ == "__main__":
    yaml = YAML(typ="safe")
    with open("icm_api.yaml", "r") as f:
        yml = yaml.load(f)

    write_views(yml)
    write_models(yml)
