from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('forecast/', views.forecast, name='forecast'),
    path('api/forecast/', views.forecast_api, name='forecast_api'),
    path('api/add_user/', views.add_user, name='add_user'),
    path('api/get_users/', views.get_users, name='get_users'),
    path('api/create_sample_users/', views.create_sample_users, name='create_users'),
   

]

