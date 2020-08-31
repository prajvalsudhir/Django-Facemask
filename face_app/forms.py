from django import forms
from .models import face_photos

class input_form(forms.ModelForm):

    class Meta():
        model = face_photos
        fields = ['img_input',]
