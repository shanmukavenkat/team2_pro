from django.db import models

# Create your models here.


from django.db import models
from django.contrib.auth.models import User

from django.db import models

class Applicant(models.Model):
    hall_ticket_no = models.CharField(max_length=100)
    rank = models.IntegerField()
    applicant_name = models.CharField(max_length=200)
    gender = models.CharField(max_length=10)
    caste = models.CharField(max_length=50)
    region = models.CharField(max_length=50)
    allocated_category = models.CharField(max_length=50)
    phase = models.CharField(max_length=20)
    group = models.CharField(max_length=20)
    college = models.CharField(max_length=100)

    def __str__(self):
        return self.applicant_name



# models.py

from django.db import models

class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(help_text="user_name", max_length=50)
    user_email = models.EmailField(help_text="user_email")
    user_password = models.CharField(help_text="user_password", max_length=50)  # Password should be hashed
    Date_Time = models.DateTimeField(auto_now=True, null=True)

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
