
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from .models import lstm

# Create your views here.
class Requestview(TemplateView):
    template_name = "request.html"
class SitePredictView(TemplateView):
    template_name = "predict.html"
def predictor(request):
    code_stock = str(request.POST['name_stock'])
    object_list = lstm(code_stock)
    return render(request, 'predict.html', {
        'object_list': object_list,
        'code_stock': code_stock,
        })
    


"""
def get_stock(request):
    if request.method == 'POST':
        symbol_stock = request.POST['name_stock']
    return symbol_stock


"""