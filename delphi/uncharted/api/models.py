import uuid
from django.db import models

class ICMMetadata(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    icmProvider = models.CharField(max_length=100, default="DUMMY")
    title = models.CharField(max_length=100, default="DelphiModel")
    version = models.IntegerField()
    created = models.DateTimeField(auto_now=True)
