from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
import requests
from django.conf import settings
from .models import Conversation
from django.views.decorators.csrf import csrf_exempt
from chatbotapp.models import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


from django.shortcuts import render
from django.http import JsonResponse
import joblib
import json
import numpy as np

# Load trained model and vectorizer
model = joblib.load("logistic_regression_chatbot_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def chatbot_view(request):
    return render(request, 'chatbot.html')

def chatbot_response(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get("message", "")

        # Transform user input using TF-IDF
        input_vector = vectorizer.transform([user_input])

        # Predict response category
        prediction = model.predict(input_vector)[0]

        response = f"The predicted category is: {prediction}"

        return JsonResponse({"response": response})
    
    return JsonResponse({"error": "Invalid request"}, status=400)



# Create your views here.
def user_index(request):
    return render(request, 'index.html')

def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    if req.method == "POST":
        user_name = req.POST.get("username")
        user_age = req.POST.get("age")
        user_phone = req.POST.get("mobile_number")
        user_email = req.POST.get("email")
        user_password = req.POST.get("Password")
        user_address = req.POST.get("address")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email = user_email
        user.user_password = user_password

        if len(req.FILES) != 0:
            image = req.FILES["profilepic"]
            user.user_image = image
            user.save()
            messages.success(req, "Updated Successfully.")
        else:
            user.save()
            messages.success(req, "Updated Successfully.")

    context = {"i": user}
    return render(req, 'app-profile.html', context)

def user_calender(request):
    return render(request, 'app-calender.html')

def user_feedback(request):
        id = request.session["user_id"]
        user = UserModel.objects.get(user_id=id)
        
        if request.method == "POST":
            rating = request.POST.get("rating")
            review = request.POST.get("review")
            
            # Sentiment analysis
            sid = SentimentIntensityAnalyzer()
            score = sid.polarity_scores(review)
            
            if score["compound"] > 0 and score["compound"] <= 0.5:
                sentiment = "positive"
            elif score["compound"] >= 0.5:
                sentiment = "very positive"
            elif score["compound"] < -0.5:
                sentiment = "negative"
            elif score["compound"] < 0 and score["compound"] >= -0.5:
                sentiment = "very negative"
            else:
                sentiment = "neutral"
            
            # Create the feedback
            Feedback.objects.create(
                Rating=rating,
                Review=review,
                Sentiment=sentiment,
                Reviewer=user
            )
            
            messages.success(request, "Feedback recorded")
            return redirect("feedback")  # Redirecting to the same page

        return render(request, 'feedback-inbox.html')

def user_feedback_graph(request):
    positive = Feedback.objects.filter(Sentiment="positive").count()
    very_positive = Feedback.objects.filter(Sentiment="very positive").count()
    negative = Feedback.objects.filter(Sentiment="negative").count()
    very_negative = Feedback.objects.filter(Sentiment="very negative").count()
    neutral = Feedback.objects.filter(Sentiment="neutral").count()
    context = {
        "vp": very_positive,
        "p": positive,
        "neg": negative,
        "vn": very_negative,
        "ne": neutral,
    }
    return render(request, 'feedback-graph.html',context)





# def user_information(request):
#     return render(request, 'information.html')

def user_feature(request):
    return render(request, 'feature.html')

def user_new_feature(request):
    return render(request, 'new-feature.html')

import random
def user_register(req):
    if req.method == "POST":
        fullname = req.POST.get("username")
        email = req.POST.get("email")
        password = req.POST.get("password")
        age = req.POST.get("age")
        address = req.POST.get("address")
        phone = req.POST.get("contact number")
        image = req.FILES.get("image") 
        
  
        missing_fields = []
        if not fullname:
            missing_fields.append("Username")
        if not email:
            missing_fields.append("Email")
        if not password:
            missing_fields.append("Password")
        if not age:
            missing_fields.append("Age")
        if not address:
            missing_fields.append("Address")
        if not phone:
            missing_fields.append("Phone Number")
        if not image:
            missing_fields.append("Profile Picture")
        
        if missing_fields:
            missing_fields_str = ", ".join(missing_fields)
            messages.warning(req, f"Please fill the following fields: {missing_fields_str}")
            return redirect("register")
        
        # Check if the email is already registered
        try:
            data = UserModel.objects.get(user_email=email)
            messages.warning(req, "Email was already registered, choose another email..!")
            return redirect("register")
        except UserModel.DoesNotExist:
            # If the email is not registered, continue with registration
            number = random.randint(1000, 9999)
            print(f"Generated OTP: {number}")  # Print OTP to the terminal
            UserModel.objects.create(
                user_name=fullname,
                user_email=email,
                user_contact=phone,
                user_age=age,
                user_password=password,
                user_address=address,
                user_image=image,
                Otp_Num=number,
            )
            mail_message = f"Registration Successfully\n Your 4 digit Pin is below\n {number}"
            send_mail("User Password", mail_message, settings.EMAIL_HOST_USER, [email])
            req.session["user_email"] = email
            messages.success(req, "Your account was created..")
            return redirect("otp")


    return render(req, 'page-register.html')



def user_login(req):
    if req.method == "POST":
        user_email = req.POST.get("email")
        user_password = req.POST.get("password")
        
        # Check for missing fields
        if not user_email or not user_password:
            messages.warning(req, "Please fill in both Email and Password.")
            return redirect("login")
        
        print(user_email, user_password)

        try:
            users_data = UserModel.objects.filter(user_email=user_email)
            if not users_data.exists():
                messages.error(req, "User does not exist.")
                return redirect("login")

            for user_data in users_data:
                if user_data.user_password == user_password:
                    if (
                        # user_data.Otp_Status == "verified"
                        user_data.user_password == user_password
                    ):
                        req.session["user_id"] = user_data.user_id
                        messages.success(req, "You are logged in.")
                        user_data.No_Of_Times_Login += 1
                        user_data.save()
                        return redirect("index")
                    elif (
                        user_data.Otp_Status == "verified"
                    ):
                        messages.info(req, "Your status is pending.")
                        return redirect("login")
                    #Go to Admin All Users overthere chnage the status to accept to override this condition.
                    elif (user_data.Otp_Status == "verified"):
                        messages.info(req, "Your Account Has been Suspended")
                        return redirect("login")
                    else:
                        messages.warning(req, "Please verify your OTP.")
                        req.session["user_email"] = user_data.user_email
                        return redirect("otp")
                else:
                    messages.error(req, "Incorrect credentials.")
                    return redirect("login")

            # Handle the case where no user data matched the password
            messages.error(req, "Incorrect credentials.")
            return redirect("user_login")
        except Exception as e:
            print(e)
            messages.error(req, "An error occurred. Please try again later.")
            return redirect("user_login")

    return render(req, 'page-login.html')

def user_screen_lock(request):
    return render(request, 'page-lock-screen.html')



def otp(req):
    user_email = req.session.get("user_email")
    if user_email:
        try:
            user_o = UserModel.objects.get(user_email=user_email)
        except UserModel.DoesNotExist:
            messages.error(req, "User not found.")
            return redirect("login")

        if req.method == "POST":
            otp1 = req.POST.get("otp1", "")
            otp2 = req.POST.get("otp2", "")
            otp3 = req.POST.get("otp3", "")
            otp4 = req.POST.get("otp4", "")

            # Check for missing OTP digits
            if not otp1 or not otp2 or not otp3 or not otp4:
                messages.error(req, "Please enter all OTP digits.")
                return redirect("otp")

            user_otp = otp1 + otp2 + otp3 + otp4
            if user_otp.isdigit():
                u_otp = int(user_otp)
                if u_otp == user_o.Otp_Num:
                    user_o.Otp_Status = "verified"
                    user_o.save()
                    messages.success(req, "OTP verification was successful. You can now login.")
                    return redirect("user_login")
                else:
                    messages.error(req, "Invalid OTP. Please enter the correct OTP.")
            else:
                messages.error(req, "Invalid OTP format. Please enter numbers only.")
                
    else:
        messages.error(req, "Session expired. Please retry the OTP verification.")

    return render(req, "main/Otp.html")

import re
import requests
from django.conf import settings
from django.shortcuts import render, redirect
from .models import Conversation

@csrf_exempt
def user_chatbot(request):
    conversations = Conversation.objects.all().order_by('created_at')
    
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if user_message:
            # Call Perplexity API
            headers = {
                "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers
            )
            
            bot_response = "Error: Could not get response from AI"
            if response.status_code == 200:
                try:
                    bot_response = response.json()['choices'][0]['message']['content']
                    
                    # Remove markdown bold (**) and any references (e.g., [1], [2], etc.)
                    bot_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', bot_response)  # Remove bold
                    bot_response = re.sub(r'\[\d+\]', '', bot_response)  # Remove reference numbers
                except:
                    pass
                
            Conversation.objects.create(
                user_message=user_message,
                bot_response=bot_response
            )
            
            return redirect('chatbot')
    
    return render(request, 'chatbot.html', {'conversations': conversations})
import nltk
import pandas as pd
from nltk import download
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer
# views.py
import requests
from django.shortcuts import render
from django.http import HttpResponse
from .models import Job
from datetime import datetime, timedelta
from django.utils import timezone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def job_list(request):
    jobs = Job.objects.filter(publication_date__gte=timezone.now() - timedelta(days=1)).order_by('-publication_date')
    return render(request, 'new-feature-list.html', {'jobs': jobs})

def job_detail(request, job_id):
    job = Job.objects.get(id=job_id)
    return render(request, 'new-feature.html', {'job': job})

import nltk

from nltk import download
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer
        
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

wnl = WordNetLemmatizer()
df_jd=pd.read_csv('dataset/training.csv')
df_jd.dropna(subset=['job_description'], inplace=True)
df_jd['job_description'] = df_jd['job_description'].astype(str)

vectorizer1 = TfidfVectorizer(ngram_range=(1, 2))
job_Description = vectorizer1.fit_transform(df_jd['job_description'])




def preprocess_text(text, wnl):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = []
    stop_words = set(stopwords.words("english"))
    for sent in sentences:
        if any(criteria in sent for criteria in ['skills', 'education']):
            words = word_tokenize(sent)
            words = [word for word in words if word not in stop_words]
            tagged_words = pos_tag(words)
            filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
            features.append(" ".join(filtered_words))
    return " ".join(features)  

import fitz 
def calculate_resume_score(resume_path):
    try:
        with fitz.open(resume_path) as doc:
            resume_text = ""
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                resume_text += page.get_text()
        resume_score = len(resume_text.split())
        max_score = 1000
        scaled_score = (resume_score / max_score) * 100
        return scaled_score
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import pandas as pd
from django.contrib import messages
import urllib.request
import urllib.parse
import random
from django.conf import settings
import os
from django.core.mail import send_mail
from django.utils.datastructures import MultiValueDictKeyError
import pandas as pd

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text


def job(request):
    top_job_descriptions = ""
    resume_score = ''
    matched_percentage = 0 
    if request.method == 'POST' and request.FILES.get('pdf-fileup'):
        uploaded_file = request.FILES['pdf-fileup']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        temp_file_path = fs.path(filename)  
        print(temp_file_path, "path is here")
        resume_score = calculate_resume_score(temp_file_path)
        # Example scenario where resume_score might be None
        resume_score = None

        # Check if resume_score is not None before rounding
        if resume_score is not None:
            resume_score = round(resume_score, 1)
        else:
            print("resume_score is None. Cannot round.")
        request.session['resumepath'] = temp_file_path
        dummy = extract_text_from_pdf(temp_file_path)
        text12 = preprocess_text(dummy, wnl)
        text13 = vectorizer1.transform([text12])
        RAM = cosine_similarity(text13, job_Description).flatten()
        
        max_similarity = max(RAM)
        matched_percentage = max_similarity * 100
        
        top_job_descriptions_df = df_jd.iloc[np.argsort(RAM)[-5:][::-1]]
        top_job_descriptions = top_job_descriptions_df.to_dict('records')
        print(top_job_descriptions,"hallooooooooooooo")
       
    return render(request, "feature.html", {'top_job_descriptions': top_job_descriptions, 'resume_score': resume_score, 'matched_percentage': matched_percentage})






import csv
from django.shortcuts import render
import random

def user_information(request):
    data = []
    with open('Apcollegslist.csv', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    
    return render(request, 'information.html', {'data': data})
