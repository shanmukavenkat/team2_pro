"""
URL configuration for chatbot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from chatbotapp import views as user_views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('user/index', user_views.user_index, name="index"),
    path('user/profile', user_views.user_profile, name="profile"),
    path('user/calender', user_views.user_calender, name="calender"),
    path('user/feedback', user_views.user_feedback, name="feedback"),
    path('user/feedback-graph', user_views.user_feedback_graph, name="feedback_graph"),
    path('user/chatbot', user_views.user_chatbot, name="chatbot"),
    path('user/information', user_views.user_information, name="information"),
    path('user/features', user_views.user_feature, name="features"),
    path('user/new-features', user_views.user_new_feature, name="new_features"),
    path('user/register', user_views.user_register, name="register"),
    path('user/otp', user_views.otp, name="otp"),
    path('', user_views.user_login, name="login"),
    path('user/lock-screen', user_views.user_screen_lock, name="lock_screen"),
    path('jobs/user/', user_views.job_list, name='job_list'),
    path('job/<int:job_id>/', user_views.job_detail, name='job_detail'),
    path("user/job/",user_views.job,name="job"),
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
