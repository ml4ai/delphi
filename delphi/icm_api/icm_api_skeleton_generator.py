""" Script to generate the skeleton of a Flask REST API from an OpenAPI 3.0
YAML specification. """

from ruamel.yaml import YAML
from typing import List
from pprint import pprint

MODELS_PREAMBLE = '''\
import json
from uuid import uuid4
from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from flask_sqlalchemy import SQLAlchemy
from delphi.icm_api import db
from sqlalchemy import PickleType
from sqlalchemy.inspection import inspect
from sqlalchemy.ext import mutable
from sqlalchemy.sql import operators
from sqlalchemy.types import TypeDecorator

class JsonEncodedList(db.TypeDecorator):
    """Enables list storage by encoding and decoding on the fly."""
    impl = db.Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return '[]'
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        else:
            return json.loads(value)

mutable.MutableList.associate_with(JsonEncodedList)

class JsonEncodedDict(db.TypeDecorator):
    """Enables JsonEncodedDict storage by encoding and decoding on the fly."""
    impl = db.Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return '{}'
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        else:
            return json.loads(value)

mutable.MutableDict.associate_with(JsonEncodedDict)

class Serializable(object):
    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]


class DelphiModel(db.Model):
    __tablename__ = "delphimodel"
    id = db.Column(db.String, primary_key=True)
    icm_metadata=db.relationship('ICMMetadata', backref='delphimodel',
                                 lazy=True, uselist=False)
    model = db.Column(db.PickleType)
'''


def get_ref(ref_string):
    return ref_string.split("/")[-1]


class Property(object):
    def __init__(
        self,
        schema_name,
        property_name,
        dictionary,
        global_schema_dict,
        is_required: bool,
    ):
        self.schema_name = schema_name
        self.name = property_name
        self.dictionary = dictionary
        self.is_required = is_required
        self.global_schema_dict = global_schema_dict
        self.property_type = self.dictionary.get("type")
        self.args = []
        if self.name == "id":
            self.args.append("primary_key = True")
            if self.dictionary.get("format") == "uuid":
                self.args.append("default = str(uuid4())")

        if not is_required:
            self.args.append("nullable = True")

        if self.property_type is not None:
            self.linetype = "Column"
            self.property_type = {
                "object": "JsonEncodedDict",
                "array": "JsonEncodedList",
            }.get(self.property_type, f"db.{self.property_type.capitalize()}")
            self.args.insert(0, self.property_type)

        else:
            self.ref = get_ref(self.dictionary.get("$ref", ""))
            if self.global_schema_dict[self.ref].get("enum") is not None:
                self.linetype = "Column"
                self.enum_values = [
                    f'"{v}"' for v in self.global_schema_dict[self.ref]["enum"]
                ]
                self.args.insert(0, f"db.Enum({', '.join(self.enum_values)})")
            else:
                self.linetype = "relationship"
                self.id_type = self.global_schema_dict[self.ref]["properties"][
                    "id"
                ]["type"]
                self.id_args = [
                    f"db.{self.id_type.capitalize()}",
                    f"db.ForeignKey('{self.ref.lower()}.id'",
                ]
                self.args = [f'"{self.ref}"']
                self.args.append(f"foreign_keys = [{self.name}_id]")
                # self.args.append("nullable=True")
                # self.args.append(f'backref = "{self.schema_name.lower()}"')

        self.property_line = (
            f'{self.name} = db.{self.linetype}({", ".join(self.args)})'
        )

    def __repr__(self):
        if self.linetype == "relationship":
            self.property_line = "\n    ".join(
                [
                    f"{self.name}_id = db.Column({', '.join(self.id_args)}), nullable=True)",
                    self.property_line,
                ]
            )
        return self.property_line


class DatabaseModel(object):
    def __init__(self, schema_name, schema_dict, global_schema_dict):
        self.schema_name = schema_name
        self.schema_dict = schema_dict
        self.global_schema_dict = global_schema_dict
        self.type = self.schema_dict.get("type")
        if self.schema_dict.get("properties") is not None:
            self.properties = self.schema_dict["properties"]
        elif self.schema_dict.get("allOf") is not None:
            self.properties = self.schema_dict["allOf"][1]["properties"]
        else:
            self.properties = {"value": self.schema_dict}

        self.required_properties = schema_dict.get("required", [])

        self.tablename = schema_name.lower()
        self.polymorphy_annotation = f"    __mapper_args__ = {{'polymorphic_identity':'{self.tablename}'"
        placeholder = f"Placeholder docstring for class {self.schema_name}."
        self.docstring = (
            f'    """ {self.schema_dict.get("description", placeholder)} """\n'
        )

        if self.schema_dict.get("allOf") is not None:
            self.parent = get_ref(self.schema_dict["allOf"][0]["$ref"])
            self.superclasses = [self.parent]
        else:
            if self.schema_dict["properties"].get("baseType") is not None:
                self.polymorphy_annotation += ", 'polymorphic_on': baseType"
            self.parent = None
            self.superclasses = ["db.Model", "Serializable"]

        self.class_declaration = (
            f"\n\nclass {schema_name}" + f"({', '.join(self.superclasses)}):"
        )

        self.properties = [
            self.process_property(
                property_name,
                property_dict,
                property_name in self.required_properties,
            )
            for property_name, property_dict in self.properties.items()
        ]

        self.polymorphy_annotation += "}"

    def process_property(self, property_name, property_dict, required):
        return Property(
            self.schema_name,
            property_name,
            property_dict,
            self.global_schema_dict,
            required,
        )

    def __repr__(self):
        lines = [self.class_declaration, self.docstring]
        if self.parent is None:
            lines.append(f'    __tablename__ = "{self.tablename}"')
        else:
            lines.append(
                f'    baseType = db.Column(db.String, default="{self.schema_name}")'
            )
        for p in self.properties:
            lines.append(f"    {p}")
        if self.schema_name in [
            "ICMMetadata",
            "CausalVariable",
            "CausalRelationship",
        ]:
            lines.append(
                "    model_id = db.Column(db.String,"
                "db.ForeignKey('delphimodel.id'))"
            )

        lines.append(self.polymorphy_annotation)
        return "\n".join(lines)


class DataclassModel(object):
    def __init__(self, schema_name, schema_dict):
        self.decorator
        self.schema_name = schema_name

    def __repr__(self):
        lines = []
        lines.append("@dataclass")
        lines.append(f"class {schema_name})")
        return "\n".join(lines)


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
        lines.append("    return '', 415")
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

    schemas = yml["components"]["schemas"]
    db_models = [
        DatabaseModel(k, v, schemas)
        for k, v in schemas.items()
        if v.get("enum") is None
        and v.get("properties") is not None
        and v.get("properties").get("id") is not None
    ]
    other_models = []
    with open("models.py", "w") as f:
        f.write(MODELS_PREAMBLE)
        for db_model in db_models:
            f.write(str(db_model))
    # write_views(yml)
    # write_models(yml)
