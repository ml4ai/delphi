from django.urls import path, re_path, include
from rest_framework.routers import DefaultRouter


from . import views

router=DefaultRouter()
router.register(r'icm', views.icm)

urlpatterns = [
    re_path(r'^', include(router.urls)),
    # path('icm/', views.icm.as_view({'get':'list'})),
    # path('icm/<uuid>/', views.icm_uuid.as_view),
    # path('icm/<uuid>/primitive', views.ICM_UUID),
    # path('icm/<uuid>/primitive', views.icm_uuid_primitive),
]
