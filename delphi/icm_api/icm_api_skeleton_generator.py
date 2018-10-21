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
    parents_list = []

    def process_properties(
        schema_name, schema, class_lines, class_declaration, parents
    ):
        class_lines.append("\n\n")
        class_lines.append(class_declaration)
        placeholder = f"Placeholder docstring for class {schema_name}."
        docstring = f"{schema.get('description', placeholder)}"
        class_lines.append(f'    """ {docstring} """\n')
        class_lines.append(f'    __tablename__ = "{schema_name}"'.lower())
        properties = schema["properties"]
        required_properties = schema.get("required", [])

        if schema_name == "ICMMetadata":
            class_lines.append("    model_id = db.Column(db.String(120),"
                        "db.ForeignKey('delphimodel.id'))")
        if parents is not None:
            foreign_key = (f"    {parents}_id = db.Column(db.Integer,"
                         + f"ForeignKey('{parents}.id'), primary_key=True)")
            class_lines.append(foreign_key)

        for index, property in enumerate(
            sorted(properties, key=lambda x: x not in required_properties)
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
                "object": "Object",
            }
            type_annotation = mapping.get(property_type, property_type)

            # ------------------------------------------------------------
            if type_annotation == "List":
                property_type = properties[property]["items"]["type"]
                type_annotation = (
                    # f"List[{mapping.get(property_type, property_type)}]"
                    f"{mapping.get(property_type, property_type)}"
                )

            # TODO - Parse dependencies so that inherited properties are
            # duplicated for child classes.

            # baseType becomes the table name
            if (
                property != "baseType"
                and property.lower() != parents
                and property.lower() not in parents_list
            ):

                if (
                    type_annotation == "db.String"
                    or type_annotation == "db.Integer"
                    or type_annotation == "db.Boolean"
                    or type_annotation == "db.Float"
                ):
                    if type_annotation == "db.String":
                        type_annotation += f"(120)"
                else:
                    type_annotation = "db.Text"

                if properties[property].get("default") is not None:
                    default_value = f"{properties[property]['default']}"
                    type_annotation += f", default={default_value}"

                if index == 0 or property == "id":
                    kwarg_2 = "primary_key=True"
                else:
                    kwarg_2 = "unique=False"

                class_lines.append(
                    f"    {property} = db.Column({type_annotation}, {kwarg_2})"
                )
            else:
                # guarantee that each table has a primary key column
                if len(properties) == 1:
                    class_lines.append(
                        "    id = db.Column(db.Integer, primary_key = True)"
                    )
                elif property.lower() in parents_list:
                    class_lines.append(
                        f"    {property.lower()}_id = db.Column(db.Integer,"
                        f"ForeignKey('{property.lower()}.id'))"
                    )

    def to_class(schema_name, schema):
        parents = None
        class_declaration = f"class {schema_name}(db.Model, Serializable):"

        class_lines = []
        if schema.get("type") == "object":

            process_properties(
                schema_name, schema, class_lines, class_declaration, parents
            )

        elif schema.get("allOf") is not None:
            parents = [
                item["$ref"].split("/")[-1]
                for item in schema["allOf"]
                if item.get("$ref")
            ]
            schema = schema["allOf"][1]
            parents = f"{','.join(parents)}"
            parents = parents.lower()
            # class_declaration = f"class {schema_name}({','.join(parents)}):"

            parents_list.append(parents)

            process_properties(
                schema_name, schema, class_lines, class_declaration, parents
            )

        elif "enum" in schema:
            class_lines.append("\n\n@unique")
            class_lines.append(f"class {schema_name}(Enum):")
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
