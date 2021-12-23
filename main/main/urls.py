"""main URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.http import request
from django.urls import path, include
from LSTM import views
from profiles import views as profiles_views



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.Requestview.as_view(), name ='request'),
    #path('accounts/', include('django.contrib.auth.urls')),
    path('logout_ok/', profiles_views.SiteLogoutOkView.as_view(), name ='logout_ok'),
    path('accounts/profile/', profiles_views.SiteProfileView.as_view(), name ='profile'),
    path('register/', profiles_views.SiteRegisterView.as_view(), name ='register'),
    path('register_ok/', profiles_views.SiteRegisterOkView.as_view(), name ='register_ok'),
    path('login/', profiles_views.SiteLoginView.as_view(), name ='login'),
    path('predict/', views.SitePredictView.as_view(), name ='predict'),
    path('predictor', views.predictor, name='predictor'),
    #path('predict.html', views.predict, name='predict'),
]
