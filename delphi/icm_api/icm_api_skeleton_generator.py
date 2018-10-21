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
from dataclasses import dataclass, field, asdict
from sqlalchemy import Table, Column, Integer, String, ForeignKey, PickleType
from flask_sqlalchemy import SQLAlchemy
from delphi.icm_api import db
from sqlalchemy.inspection import inspect

class Serializable(object):
    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]


class DelphiModel(db.Model):
    __tablename__ = "delphimodel"
    id = db.Column(db.String(120), primary_key=True)
    icm_metadata=db.relationship('ICMMetadata', backref='delphimodel',
                                 lazy=True, uselist=False)
    model = db.Column(db.PickleType)
"""
    )

    def process_properties(
        schema_name, schema, class_lines, is_db_object, parent=None
    ):
        placeholder = f"Placeholder docstring for class {schema_name}."
        docstring = f"{schema.get('description', placeholder)}"
        class_lines.append(f'    """ {docstring} """\n')
        properties = schema["properties"]
        tablename = schema_name.lower()
        if is_db_object:
            if parent is None:
                class_lines.append(f'    __tablename__ = "{tablename}"')
            if schema["properties"].get("baseType") is not None:
                polymorphy_annotation = f"    __mapper_args__ = {{'polymorphic_identity':'{tablename}'"
                polymorphy_annotation += ", 'polymorphic_on': baseType"
            polymorphy_annotation += "}"

        required_properties = schema.get("required", [])

        for property in sorted(
            properties, key=lambda x: x not in required_properties
        ):

            property_ref = properties[property].get("$ref", "").split("/")[-1]
            if property_ref != "" and property_ref not in global_schema_list:
                global_schema_list.append(property_ref)

            # if the current property does not have type, use property_ref
            # instead, so it won't be none.
            property_type = properties[property].get("type", property_ref)
            mapping = {
                "string": "db.String",
                "integer": "db.Integer",
                "None": "",
                "array": "List",
                "boolean": "db.Boolean",
                "number": "db.Float",
                "object": "db.PickleType",
            }
            type_annotation = mapping.get(property_type, "db.String")

            # ------------------------------------------------------------
            if type_annotation == "List":
                property_type = properties[property]["items"]["type"]
                type_annotation = (
                    f"{mapping.get(property_type, property_type)}"
                )

            # baseType becomes the table name
            if properties[property].get("default") is not None:
                default_value = f"{properties[property]['default']}"
                type_annotation += f", default={default_value}"

            kwargs = []

            if property == "id":
                kwargs.append("primary_key=True")
            else:
                kwargs.append("unique=False")

            class_lines.append(
                f"    {property} = db.Column({type_annotation}, {','.join(kwargs)})"
            )

        if is_db_object and parent is not None:
            class_lines.append(
                f"    id = db.Column(db.String, ForeignKey('{parent.lower()}.id'), primary_key=True)"
            )

        if schema_name in [
            "ICMMetadata",
            "CausalVariable",
            "CausalRelationship",
        ]:
            class_lines.append(
                "    model_id = db.Column(db.String(120),"
                "db.ForeignKey('delphimodel.id'))"
            )
        if is_db_object:
            class_lines.append(polymorphy_annotation)

    def to_class(schema_name, schema):
        is_db_object = False
        class_lines = ["\n"]

        if schema.get("type") == "object":
            object_type = "parent"
            if schema["properties"].get("id") is not None:
                is_db_object = True
                base = "db.Model, Serializable"
            else:
                base = "object"
            class_declaration = f"class {schema_name}({base}):"
            class_lines.append(class_declaration)
            process_properties(schema_name, schema, class_lines, is_db_object)

        elif schema.get("allOf") is not None:
            parents = [
                item["$ref"].split("/")[-1]
                for item in schema["allOf"]
                if item.get("$ref")
            ]
            if schemas[parents[0]]["properties"].get("id") is not None:
                is_db_object = True
            schema = schema["allOf"][1]
            class_declaration = f"class {schema_name}({','.join(parents)}):"
            class_lines.append(class_declaration)

            process_properties(
                schema_name, schema, class_lines, is_db_object, parents[0]
            )

        elif "enum" in schema:
            class_lines.append("@unique")
            class_declaration = f"class {schema_name}(Enum):"
            class_lines.append(class_declaration)
            for option in schema["enum"]:
                class_lines.append(f'    {option} = "{option}"')

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
            f"\n\n@bp.route('{modified_path}', methods=['{http_method.upper()}'])"
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
                "from flask import Flask, jsonify, request, Blueprint",
                "from delphi.icm_api.models import *",
                "bp = Blueprint('icm_api', __name__)",
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

    # write_views(yml)
    write_models(yml)
