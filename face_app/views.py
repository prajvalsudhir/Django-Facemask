from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import (TemplateView,
                                  ListView,DetailView,CreateView,
                                  UpdateView,DeleteView)
from django.http import HttpResponseRedirect, HttpResponse
from .face_model import predict
from .forms import input_form
from .models import face_photos,predicted_photos

from .face_model import predict

# Create your views here.

def user_input(requests):
    if requests.method == 'POST':
        form = input_form(requests.POST,requests.FILES)
        # print(form)
        if form.is_valid():
            print(form.cleaned_data['img_input'])
            form.save()
            # face_photo = face_photos()
            # face_photo.img_input = 'images_input/' + str(form.cleaned_data['img_input'])
            # face_photo.save()
            input_path = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\media\\images_input\\' + str(form.cleaned_data['img_input'])
            output_path = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\media\\images_output\\'  + str(form.cleaned_data['img_input'])
            pro_img = predict(input_path,output_path,form)
            return render(requests,'face_app/result.html',{'processed_img':pro_img})
            # return HttpResponse('image uploaded')
    else:
        form = input_form()
    return render(requests,'face_app/index.html',{'form':form})


# def show_result(DetailView):
#     model = predicted_photos


