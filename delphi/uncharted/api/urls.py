from django.urls import path

from . import views

urlpatterns = [

    # <server_address> as per above, on localhost via port 8080 

    path('icm/', views.get_delphi_model_ids, name="DelphiModels"),
    # this would call the function get_Delphi_model_ids defined in views.py
    # output would be available at http://<server_address>/icm

    # path(r'^icm/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{'
        # r'12})/primitive', name="DelphiPrimitives"),
    # example of more realistic path as per ICM API, with regex for a legitimate model uuid
    # path would be: http://<server_address>/icm/<uuid>/primitive

    # path(r'^model/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{'
        # r'12})/some_other_model_function', name="SomethingeElse")

]
