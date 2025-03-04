from django.db import models

# Create your models here.


from django.db import models
from django.contrib.auth.models import User


class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(help_text="user_name", max_length=50)
    user_age = models.IntegerField(null=True)
    user_email = models.EmailField(help_text="user_email")
    user_password = models.EmailField(help_text="user_password", max_length=50)
    user_address = models.TextField(help_text="user_address", max_length=100)
    user_subject = models.TextField(
        help_text="user_subject", max_length=100, default="default_value_here"
    )
    user_contact = models.CharField(help_text="user_contact", max_length=15, null=True)
    user_image = models.ImageField(upload_to="media/", null=True)
    Date_Time = models.DateTimeField(auto_now=True, null=True)
    Otp_Num = models.IntegerField(null=True)
    Otp_Status = models.TextField(default="pending", max_length=60, null=True)
    Last_Login_Time = models.TimeField(null=True)
    Last_Login_Date = models.DateField(auto_now_add=True, null=True)
    No_Of_Times_Login = models.IntegerField(default=0, null=True)
    Message = models.TextField(max_length=250, null=True)

    class Meta:
        db_table = "UserModel"
            

class Feedback(models.Model):
    Feed_id = models.AutoField(primary_key=True)
    Rating = models.CharField(max_length=100, null=True)
    Review = models.CharField(max_length=225, null=True)
    Sentiment = models.CharField(max_length=100, null=True)
    Reviewer = models.ForeignKey(UserModel, on_delete=models.CASCADE, null=True)
    datetime = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "feedback_details"


class Conversation(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_message[:50]}..."
    

# models.py
from django.db import models

class Job(models.Model):
    title = models.CharField(max_length=200)
    company_name = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    job_type = models.CharField(max_length=50)
    publication_date = models.DateTimeField()
    candidate_required_location = models.CharField(max_length=100)
    salary = models.CharField(max_length=100, null=True, blank=True)
    description = models.TextField()
    url = models.URLField()

    def __str__(self):
        return self.title
    
    from django.db import models


from django.db import models

from django.db import models

class College(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    course = models.CharField(max_length=255)
    fees = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return self.name
