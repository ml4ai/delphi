from rest_framework import serializers
from api.models import ICMMetadata

class ICMMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ICMMetadata
        fields = '__all__'
