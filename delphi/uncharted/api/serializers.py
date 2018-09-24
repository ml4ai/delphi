from rest_framework import serializers

from api.models import DelphiModel


class DelphiModelSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    title = serializers.CharField(
        required=False, allow_blank=True, max_length=100
    )

    def create(self, validated_data):
        """ Create and return a new `DelphiModel` instance, given the validated data. """
        return DelphiModel.objects.create(**validated_data)

    def update(self, instance, validated_data)
