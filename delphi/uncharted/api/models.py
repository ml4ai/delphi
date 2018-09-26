import uuid
from django.db import models

class CausalVariable(models.Model):
    basetype = models.CharField(max_length=30, default="string")
    editable = models.BooleanField(default=True)
    id = models.CharField(max_length=100, default="", primary_key=True)
    label = models.CharField(max_length=100, default="")
    description = models.CharField(max_length=300, default="")
    lastUpdated = models.DateTimeField(auto_now=True)
    units = models.CharField(max_length=100, default="")

class ICMMetadata(models.Model):
    id = models.UUIDField(primary_key=True, editable=False)
    icmProvider = models.CharField(max_length=100, default="UAZ")
    title = models.CharField(max_length=100, default="DelphiModel")
    version = models.IntegerField(default=1)
    created = models.DateTimeField(auto_now=True)
