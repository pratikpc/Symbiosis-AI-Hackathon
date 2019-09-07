#!/usr/bin/env python3.6

from speech_to_text import SpeechToText
from predict_model import PredictResults
import sys
import os

# Get the full path from the given directory
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == "__main__":
    path = "/App/audio"
    if len(sys.argv) == 2:
        path = sys.argv[1]

    print("Path is", path)

    files = listdir_fullpath(path)

    with open("/App/out/vortex-I-man01.csv", "w") as f:
        for file in files:
            # Convert Speech to Text
            print("Converting Speech to Text for ", file)
            text = SpeechToText(file)
            # text = path
            # file = "abc.wav"

            # Use this to predict results
            print("Text Recognised is ", text)
            prediction = PredictResults(text)

            f.write(file + ", " + prediction)
            
            print(prediction)    
