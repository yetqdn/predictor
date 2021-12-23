
from typing import Reversible
from django.contrib.auth import forms
from django.forms import widgets
from django.shortcuts import redirect
from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.forms import UserCreationForm, UsernameField
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.contrib.auth.views import LoginView, LogoutView

# Create your views here.
class SiteLoginView(LoginView):
    template_name = 'login.html'
class SiteLogoutOkView(LogoutView):
    template_name = 'logout_ok.html'
class SiteProfileView(LoginRequiredMixin, TemplateView):
    template_name = 'profile.html'
class SiteRegisterOkView(TemplateView):
    template_name = 'register_ok.html'

class RegisterForm(UserCreationForm):
    
    class Meta:
        model = User
        fields = ('username', 'email',)
        fields_classes = {'username': UsernameField}
        

class SiteRegisterView(FormView):
    template_name = 'register.html'
    form_class = RegisterForm

    def form_valid(self, form):
        data = form.cleaned_data
        new_user = User.objects.create_user(
            username=data['username'],
            password=data['password2'],
            email=data['email']
        )
        from pprint import pprint; pprint(data)
        return redirect('register_ok')
        """url = f"{reverse('register_ok')}?username={new_user.username}"
        from pprint import pprint; pprint(url)
        return redirect(url)"""

