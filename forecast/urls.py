from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('forecast/', views.forecast, name='forecast'),
    path('api/forecast/', views.forecast_api, name='forecast_api'),


]

