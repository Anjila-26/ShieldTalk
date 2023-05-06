import cv2
import os
import tensorflow as tf
import whisper


def capture_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    frameNr = 0
    while(True):
        ret, frame = capture.read() 
        if ret == False:
            break;
        if frameNr % 8 == 0:  
            cv2.imwrite(f'C:/Users/anjil/Videos/Deerhack/Anjila/Anjila_{frameNr}.jpg', frame)
        frameNr = frameNr + 1 
    capture.release() 

def delete_video():
    file_path = 'static/Video1.mp4'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been deleted.')
    else:
        print(f'{file_path} does not exist.')

def delete_audio():
    file_path = 'static/Video1.mp3'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been deleted.')
    else:
        print(f'{file_path} does not exist.')

def delete_picture():
    file_path = 'api/static/api/media/barchart.png'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been deleted.')
    else:
        print(f'{file_path} does not exist.')


def findLen(str):
    counter = 0   
    for i in str:
        counter += 1
    return counter

model = whisper.load_model("base")








