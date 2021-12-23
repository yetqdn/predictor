from django import forms

class Form_request(forms.Form):
    your_stock = forms.CharField(label='your_stock', max_length=100)