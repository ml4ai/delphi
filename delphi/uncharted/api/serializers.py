from rest_framework import serializers
from api.models import DelphiModel

class DelphiModelSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200)
    dateCreated = serializers.DateTimeField()
    delta_t = serializers.FloatField()
