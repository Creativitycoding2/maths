from django.urls import path
from . import views


urlpatterns = [
    path('', views.num1, name="num1"),
    path('result/', views.num2, name="num2")
]
