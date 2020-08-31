from django.db import models

# Create your models here.

class face_photos(models.Model):
    img_input = models.ImageField(upload_to='images_input/')
    # img_output = models.ImageField(upload_to='images_output/',default='default.png',blank=True,null=True)


    # def __str__(self):
    #     name = self.clean()
    #     return name

class predicted_photos(models.Model):
    img_output = models.ImageField(upload_to='images_output/',default='default.png',blank=True,null=True)
