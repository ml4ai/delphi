from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET
from rest_framework.decorators import action
from rest_framework.response import Response
from delphi.AnalysisGraph import AnalysisGraph
from rest_framework.serializers import ListSerializer, BaseSerializer
from delphi.export import to_dict
from rest_framework.viewsets import ModelViewSet, ReadOnlyModelViewSet

from api.serializers import *

from os import environ
from os.path import join

PORT_NUMBER = environ.get("PORT", 8080)
HOST_ID = environ.get('HOST_ID', 'localhost')

list_of_icm_ids = [1, 2, 3, 4]

G = AnalysisGraph.from_pickle('../../delphi_cag.pkl')
model_dict = {'1': G, '2': G, '3': G, '4': G}

class icm(ReadOnlyModelViewSet):
    queryset = ICMMetadata.objects.all()
    serializer_class = ICMMetadataSerializer
    http_method_names = ['get']

    def list(self, request):
        return Response(ICMMetadata.objects.values_list('id', flat=True))

# @require_GET
# def icm_uuid(request, uuid):
    # model = model_dict[uuid]
    # icm_metadata = ICMMetadata(id = model.id)
    # serializer = ICMMetadataSerializer(icm_metadata)
    # return JsonResponse(serializer.data)

@require_GET
def icm_uuid_primitive(request, uuid):
    return JsonResponse(list(G.nodes)+list(G.edges), safe=False)

if __name__=='__main__':
    print('ok')
