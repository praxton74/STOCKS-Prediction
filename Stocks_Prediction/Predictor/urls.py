# predictor/urls.py

from django.urls import path
from . import views


urlpatterns = [
    path('', views.predict_price, name='predict_price'),
    path('predict_price_api/', views.predict_price_api, name='predict_price_api'),
]
