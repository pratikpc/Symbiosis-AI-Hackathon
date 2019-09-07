#from speech_to_text import SpeechToText
from predict_model import PredictResults
import sys

file_name = sys.argv[1]

#text = SpeechToText(file_name)
prediction = PredictResults(file_name)
print(prediction)