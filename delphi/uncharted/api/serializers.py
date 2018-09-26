from rest_framework import serializers
from api.models import *

class ICMMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ICMMetadata
        fields = '__all__'

class CausalVariableSerializer(serializers.ModelSerializer):
    class Meta:
        model = CausalVariable
        fields = '__all__'
