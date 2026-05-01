from django.urls import path
from . import views

urlpatterns = [
    path('getJSonValue', views.get_json_value, name='get_json_value'),
]