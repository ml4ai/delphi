from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from delphi.icm_api import db

@unique
class ICMProvider(Enum):
    BAE = "BAE"
    BBN = "BBN"
    STR = "STR"
    DUMMY = "DUMMY"


@unique
class LifecycleState(Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    EXPERIMENTAL = "EXPERIMENTAL"
    OPERATIONAL = "OPERATIONAL"
    SUSPENDED = "SUSPENDED"
    ARCHIVED = "ARCHIVED"
    CREATED = "CREATED"



class User(db.Model):
    """Placeholder docstring for class User. """

    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=False)
    firstName = db.Column(db.String(120), unique=False)
    lastName = db.Column(db.String(120), unique=False)
    email = db.Column(db.String(120), unique=False)
    password = db.Column(db.String(120), unique=False)
    phone = db.Column(db.String(120), unique=False)
    userStatus = db.Column(db.Integer, unique=False)



class ICMMetadata(db.Model):
    """Placeholder docstring for class ICMMetadata. """

    __tablename__ = "icmmetadata"
    id = db.Column(db.String(120), primary_key=True)
    icmProvider = db.Column(db.Text, unique=False)
    title = db.Column(db.String(120), unique=False)
    version = db.Column(db.Integer, unique=False)
    created = db.Column(db.String(120), unique=False)
    createdByUser = db.Column(db.Text, unique=False)
    lastAccessed = db.Column(db.String(120), unique=False)
    lastAccessedByUser = db.Column(db.Text, unique=False)
    lastUpdated = db.Column(db.String(120), unique=False)
    lastUpdatedByUser = db.Column(db.Text, unique=False)
    estimatedNumberOfPrimitives = db.Column(db.Integer, unique=False)
    lifecycleState = db.Column(db.Text, unique=False)
    derivation = db.Column(db.String(120), unique=False)



class ServerResponse(db.Model):
    """Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = db.Column(db.String(120), primary_key=True)
    message = db.Column(db.String(120), unique=False)



class Range(db.Model):
    """Top level range object used in a CausalVariable """

    __tablename__ = "range"
    id = db.Column(db.Integer, primary_key = True)



class IntegerRange(db.Model):
    """The range for an integer value """

    __tablename__ = "integerrange"
    range_id = db.Column(db.Integer, ForeignKey('range.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class FloatRange(db.Model):
    """The range for a floating point value """

    __tablename__ = "floatrange"
    range_id = db.Column(db.Integer, ForeignKey('range.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class BooleanRange(db.Model):
    """Denotes a boolean range """

    __tablename__ = "booleanrange"
    range_id = db.Column(db.Integer, ForeignKey('range.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class EnumRange(db.Model):
    """The values of an enumeration """

    __tablename__ = "enumrange"
    range_id = db.Column(db.Integer, ForeignKey('range.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class DistributionEnumRange(db.Model):
    """The range of classifications that can be reported in a DistributionEnumValue """

    __tablename__ = "distributionenumrange"
    range_id = db.Column(db.Integer, ForeignKey('range.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class Value(db.Model):
    """Top level value object used in a TimeSeriesValue """

    __tablename__ = "value"
    id = db.Column(db.Integer, primary_key = True)



class IntegerValue(db.Model):
    """An integer value """

    __tablename__ = "integervalue"
    value_id = db.Column(db.Integer, ForeignKey('value.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class FloatValue(db.Model):
    """A floating point value """

    __tablename__ = "floatvalue"
    value_id = db.Column(db.Integer, ForeignKey('value.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class BooleanValue(db.Model):
    """A boolean value """

    __tablename__ = "booleanvalue"
    value_id = db.Column(db.Integer, ForeignKey('value.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class EnumValue(db.Model):
    """An enumeration value defined in the EnumRange for the CausalVariable """

    __tablename__ = "enumvalue"
    value_id = db.Column(db.Integer, ForeignKey('value.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)



class DistributionEnumValue(db.Model):
    """A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    __tablename__ = "distributionenumvalue"
    value_id = db.Column(db.Integer, ForeignKey('value.id'), primary_key=True)
    id = db.Column(db.Integer, primary_key = True)


@unique
class TimeSeriesState(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"


@unique
class TimeSeriesTrend(Enum):
    LARGE_INCREASE = "LARGE_INCREASE"
    SMALL_INCREASE = "SMALL_INCREASE"
    NO_CHANGE = "NO_CHANGE"
    SMALL_DECREASE = "SMALL_DECREASE"
    LARGE_DECREASE = "LARGE_DECREASE"



class TimeSeriesValue(db.Model):
    """Time series value at a particular time """

    __tablename__ = "timeseriesvalue"
    time = db.Column(db.String(120), primary_key=True)
    value_id = db.Column(db.Integer, ForeignKey('value.id'))
    active = db.Column(db.Text, unique=False)
    trend = db.Column(db.Text, unique=False)



class CausalPrimitive(db.Model):
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    namespaces = db.Column(db.Text, unique=False)
    types = db.Column(db.String(120), unique=False)
    editable = db.Column(db.Boolean, default=True, unique=False)
    disableable = db.Column(db.Boolean, default=True, unique=False)
    disabled = db.Column(db.Boolean, default=False, unique=False)
    id = db.Column(db.String(120), primary_key=True)
    label = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    lastUpdated = db.Column(db.String(120), unique=False)



class Entity(db.Model):
    """API definition of an entity.  """

    __tablename__ = "entity"
    causalprimitive_id = db.Column(db.Integer, ForeignKey('causalprimitive.id'), primary_key=True)
    confidence = db.Column(db.Float, primary_key=True)



class CausalVariable(db.Model):
    """API definition of a causal variable.  """

    __tablename__ = "causalvariable"
    causalprimitive_id = db.Column(db.Integer, ForeignKey('causalprimitive.id'), primary_key=True)
    range_id = db.Column(db.Integer, ForeignKey('range.id'))
    units = db.Column(db.String(120), unique=False)
    backingEntities = db.Column(db.String(120), unique=False)
    lastKnownValue = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)



class ConfigurationVariable(db.Model):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    __tablename__ = "configurationvariable"
    causalprimitive_id = db.Column(db.Integer, ForeignKey('causalprimitive.id'), primary_key=True)
    units = db.Column(db.String(120), primary_key=True)
    lastKnownValue = db.Column(db.Text, unique=False)
    range_id = db.Column(db.Integer, ForeignKey('range.id'))



class CausalRelationship(db.Model):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """

    __tablename__ = "causalrelationship"
    causalprimitive_id = db.Column(db.Integer, ForeignKey('causalprimitive.id'), primary_key=True)
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)
    strength = db.Column(db.Float, unique=False)
    reinforcement = db.Column(db.Boolean, unique=False)



class Relationship(db.Model):
    """API definition of a generic relationship between two primitives """

    __tablename__ = "relationship"
    causalprimitive_id = db.Column(db.Integer, ForeignKey('causalprimitive.id'), primary_key=True)
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)



class Evidence(db.Model):
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = db.Column(db.String(120), primary_key=True)
    link = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    category = db.Column(db.String(120), unique=False)
    rank = db.Column(db.Integer, unique=False)



class Projection(db.Model):
    """Placeholder docstring for class Projection. """

    __tablename__ = "projection"
    numSteps = db.Column(db.Integer, primary_key=True)
    stepSize = db.Column(db.String(120), unique=False)
    startTime = db.Column(db.String(120), unique=False)



class Experiment(db.Model):
    """structure used for experimentation """

    __tablename__ = "experiment"
    id = db.Column(db.String(120), primary_key=True)
    label = db.Column(db.String(120), unique=False)
    options = db.Column(db.Text, unique=False)



class ForwardProjection(db.Model):
    """a foward projection experiment """

    __tablename__ = "forwardprojection"
    experiment_id = db.Column(db.Integer, ForeignKey('experiment.id'), primary_key=True)
    interventions = db.Column(db.Text, primary_key=True)
    projection = db.Column(db.Text, unique=False)



class SensitivityAnalysis(db.Model):
    """a sensitivity analysis experiment """

    __tablename__ = "sensitivityanalysis"
    experiment_id = db.Column(db.Integer, ForeignKey('experiment.id'), primary_key=True)
    variables = db.Column(db.String(120), primary_key=True)



class ExperimentResult(db.Model):
    """Notional model of experiment results """

    __tablename__ = "experimentresult"
    id = db.Column(db.String(120), primary_key=True)



class ForwardProjectionResult(db.Model):
    """The result of a forward projection experiment """

    __tablename__ = "forwardprojectionresult"
    experimentresult_id = db.Column(db.Integer, ForeignKey('experimentresult.id'), primary_key=True)
    projection = db.Column(db.Text, primary_key=True)
    results = db.Column(db.Text, unique=False)



class SensitivityAnalysisResult(db.Model):
    """The result of a sensitivity analysis experiment """

    __tablename__ = "sensitivityanalysisresult"
    experimentresult_id = db.Column(db.Integer, ForeignKey('experimentresult.id'), primary_key=True)
    results = db.Column(db.Text, primary_key=True)



class Traversal(db.Model):
    """Placeholder docstring for class Traversal. """

    __tablename__ = "traversal"
    maxDepth = db.Column(db.Integer, primary_key=True)



class Version(db.Model):
    """Placeholder docstring for class Version. """

    __tablename__ = "version"
    icmVersion = db.Column(db.String(120), primary_key=True)
    icmProviderVersion = db.Column(db.String(120), unique=False)
