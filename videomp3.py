import whisper
import os
import subprocess
import contractions
import re
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from frame import delete_picture

train = pd.read_csv('training_data/train.csv/train.csv')
test = pd.read_csv('training_data/test.csv/test.csv')
test_y = pd.read_csv('training_data/test_labels.csv/test_labels.csv')


model = whisper.load_model("base")

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"


def translate(audio):
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio,**translate_options)
    return result

def clean_and_preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r"https?://S+www.\.\S+", "", text)
    
    # Remove HTML tags
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(html, "", text)
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

train['text_clean'] = train['comment_text'].apply(lambda x: clean_and_preprocess_text(x))

# Subsetting labels from the training data
train_labels = train[['toxic', 'severe_toxic',
                      'obscene', 'threat', 'insult', 'identity_hate']]
label_count = train_labels.sum()


X_train, X_test, y_train, y_test = train_test_split(train['text_clean'], train_labels, test_size= 0.3)

import matplotlib.pyplot as plt
import numpy as np

def barchart(prob_dict):

    # Sort the dictionary in descending order of values
    sorted_prob_dict = {k: v for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Create lists of labels and their corresponding probabilities
    labels = list(sorted_prob_dict.keys())
    probabilities = list(sorted_prob_dict.values())
 
    # Set up the bar graph
    plt.bar(labels, probabilities, color='brown')
    
    # Set the y-axis limits to 0 and 1
    plt.ylim(0, 1)
    
    # Set the x- and y-axis labels
    plt.xlabel('Label')
    plt.ylabel('Probability')
    
    # Set the title of the plot
    plt.title('Label Probabilities')
    
    # Show the plot
    plot_path = 'api/static/api/media/barchart.png'

    if os.path.exists(plot_path):
        os.remove(plot_path)

    plt.savefig(plot_path)
    

def barchart(prob_dict):
    plt.clf()
    # Sort the dictionary in descending order of values
    sorted_prob_dict = {k: v for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Create lists of labels and their corresponding probabilities
    labels = list(sorted_prob_dict.keys())
    probabilities = list(sorted_prob_dict.values())
 
    # Set up the bar graph
    plt.bar(labels, probabilities, color='brown')
    
    # Set the y-axis limits to 0 and 1
    plt.ylim(0, 1)
    
    # Set the x- and y-axis labels
    plt.xlabel('Label')
    plt.ylabel('Probability')
    
    # Set the title of the plot
    plt.title('Label Probabilities')
    
    # Show the plot
    plot_path = './api/static/api/media/barchart.png'

    if os.path.exists(plot_path):
        os.remove(plot_path)

    plt.savefig(plot_path)







