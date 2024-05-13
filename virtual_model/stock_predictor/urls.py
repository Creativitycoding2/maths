# stock_predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload'),
    path('stock_prediction/', views.stock_prediction, name='stock_prediction'),
]
