from django.urls import path
from . import views

urlpatterns = [
    path('', views.audio_detector_view, name='audio_detector'),
    
]