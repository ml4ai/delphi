from django.http import JsonResponse, HttpResponse

from os import environ
from os.path import join

PORT_NUMBER = environ.get("PORT", 8080)
HOST_ID = environ.get('HOST_ID', 'localhost')

def get_delphi_model_ids(request):
    if request.method=='GET':
        return HttpResponse('blah')
