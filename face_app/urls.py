from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls import static

urlpatterns = [
    path('',views.user_input,name='image_upload')
]

if settings.DEBUG:
    urlpatterns += static.static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)