from django.urls import path,include
from .views import *

urlpatterns = [
    path('video',Home,name='home'),
    path('frame',Button,name='button'),
    path('shieldtalk', main_page, name='main_page'),
    path('text', my_view, name='text_page'),
    path('aboutus', about_us, name = 'about_us'),
    path('why',why,name='why'),
    path('team',team,name='team')
]