from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from .forms import VideoForm,MyForm
from .models import Video
import os
import cv2
import numpy as np
from frame import *
from videomp3 import *
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
import matplotlib.pyplot as plt
from django.http import FileResponse
import pickle
with open("trained_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
model_path = "model.h5"

model = load_model(model_path)

import os

def my_view(request):
    '''
    Takes the text preprocesses it, feeds it to the model and gets prediction and visualizes that prediction
    '''
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():

            #vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
              #          stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
            
            new_text = clean_and_preprocess_text(form.cleaned_data['my_input'])
            

            # Transform the new text data using the same vectorizer object
            new_text_vect = vectorizer.transform([new_text])

            # Reshape the transformed data into the required input shape
            new_text_vect_array = new_text_vect.toarray().reshape(1, 1, -1)

            # Get predictions for the new text data
            predictions = model.predict(new_text_vect_array)

            predictions = predictions[0]

            a = {'toxic':predictions[0],
                'severe_toxic':predictions[1],	
                'obscene':predictions[2],	
                'threat':predictions[3],
                'insult':predictions[4],	
                'identity_hate':predictions[5]
                }
            
            barchart(a)

            return render(request, 'visualization.html')
        
    else:
        
        form = MyForm()
    return render(request, 'text.html', {'forms': form})



def Home(request):
    '''
    Takes the video, audio and text, preprocess it and predicts the toxicity using the model and visualizes the prediction
    '''
    # get all videos from the database
    all_videos = Video.objects.all()

    if request.method == "POST":
        # create a form instance and populate it with data from the request
        form = VideoForm(request.POST, request.FILES)
        
        # check if the form is valid
        if form.is_valid():
            # save the form to the database
            video = form.save(commit=False)
            if not video.caption:
                video.caption = 'No caption provided'
            
            video.title = 'Video 1'
            video.save()

            new_filename = 'Video1.mp4'
            old_path = video.video_file.path
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            os.rename(old_path, new_path)

            input_video = 'static/Video1.mp4'
            audio_file = video2mp3(input_video)
            result = translate(audio_file)

            #vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
                        #stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
            
            new_text = clean_and_preprocess_text(result['text'])
            

            # Transform the new text data using the same vectorizer object
            new_text_vect = vectorizer.transform([new_text])

            # Reshape the transformed data into the required input shape
            new_text_vect_array = new_text_vect.toarray().reshape(1, 1, -1)

            # Get predictions for the new text data
            predictions = model.predict(new_text_vect_array)

            predictions = predictions[0]

            a = {'toxic':predictions[0],
                'severe_toxic':predictions[1],	
                'obscene':predictions[2],	
                'threat':predictions[3],
                'insult':predictions[4],	
                'identity_hate':predictions[5]
                }
            
            barchart(a)

        
            delete_video()
            delete_audio()

            return render(request, 'visualization.html')
            # redirect to a success page
            #return render(request, "button.html")
    else:
        # create an empty form instance
        form = VideoForm()

    # render the template with the form and all the videos
    return render(request, "home.html", {"form": form, "all_videos": all_videos})

def Button(request):
    capture_frames('static/Video1.mp4')
    delete_video()
    return render(request, 'button.html')



def load_video(file_path):
    '''
    Loads and returns video after taking the file path
    :params:
    file_path(str): Takes file path
    :returns:
    f(object): Video
    '''
    with open(file_path, 'rb') as f:
        return f.read()
    

def loading_video(request): 
    '''
    Loads the video
    '''
    video_file_path = 'static/Video1.mp4'
    video_content = load_video(video_file_path)
    response = HttpResponse(content_type='video/mp4')
    response.write(video_content)
    return response

def main_page(request):
    '''
    Renders the main page
    '''
    return render(request, 'main_page.html')

def text_page(request):
    '''
    Renders the text page
    '''
    return render(request, 'text.html')

def about_us(request):
    '''
    Renders the About Us page
    '''
    return render(request, 'about_us.html')

def why(request):
    return render(request, 'why.html')

def team(request):
    return render(request, 'team.html')


            
    





    
