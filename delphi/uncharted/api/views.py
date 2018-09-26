from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET
from rest_framework.response import Response
from delphi.AnalysisGraph import AnalysisGraph
from delphi.export import to_json_dict

from api.serializers import *

from os import environ
from os.path import join

PORT_NUMBER = environ.get("PORT", 8080)
HOST_ID = environ.get('HOST_ID', 'localhost')

list_of_icm_ids = [1, 2, 3, 4]

G = AnalysisGraph.from_pickle('../../delphi_cag.pkl')
model_dict = {'1': G, '2': G, '3': G, '4': G}

@require_GET
def icm(request):
    return JsonResponse(list_of_icm_ids, safe=False)

@require_GET
def icm_uuid(request, uuid):
    model = model_dict[uuid]
    icm_metadata = ICMMetadata(id = model.id)
    serializer = ICMMetadataSerializer(icm_metadata)
    return JsonResponse(serializer.data)

@require_GET
def icm_uuid_primitive(request, uuid):
    return JsonResponse(list(G.nodes)+list(G.edges), safe=False)

if __name__=='__main__':
    print('ok')
