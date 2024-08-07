from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.home, name="home"),
    path("projects/", views.projects, name="projects"),
    path("contact/", views.contact, name="contact"),
    path("mnist/", views.mnist, name="mnist"),
    path("fashion_mnist/", views.fashion_mnist, name="fashion_mnist"),
    path("visualization/", views.visualization, name="visualization"),
]
