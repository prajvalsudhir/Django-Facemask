from django.contrib import admin
from .models import face_photos,predicted_photos
# Register your models here.

admin.site.register(face_photos)
admin.site.register(predicted_photos)