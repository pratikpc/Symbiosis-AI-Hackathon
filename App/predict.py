#!/usr/bin/env python3.6

#from speech_to_text import SpeechToText
from predict_model import PredictResults
import sys
import os

# Get the full path from the given directory
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You need to provide path as argument")
        exit()
    path = sys.argv[1]

    # print("Path is", path)

    # files = listdir_fullpath(path)

    # for file in files:
    # Convert Speech to Text
    # print("Converting Speech to Text for ", file)
    # text = SpeechToText(file)
    text = path

    # Use this to predict results
    print("Text Recognised is ", text)
    prediction = PredictResults(text)
    
    print(prediction)    
