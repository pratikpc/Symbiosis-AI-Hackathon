#!/usr/bin/env python3.6

# import nltk_load_new
from speech_to_text import SpeechToText
from predict_model import PredictResults, ModelColdStarter
from translator import TranslateToEnglish
import sys
import os
import app_utils

languages = ["English", "Hindi", "Marathi"]
if __name__ == "__main__":
    path = "./audio"
    out_path = "./out"
    if len(sys.argv) == 3:
        path = sys.argv[1]
        out_path = sys.argv[2]

    app_utils.DebugCommand("Path is", path)
    app_utils.create_fullpath_if_not_exists(out_path)

    for language in languages:
        language_path = os.path.join(path, language)
        if (not app_utils.CheckIfPathExists(language_path)):
            continue
        files = app_utils.listdir_fullpath(language_path)
        with open(os.path.join(out_path, "vortex_" + language + ".csv"), "w") as f:
            for file in files:
                # Convert Speech to Text
                print("Converting Speech to Text for ", file)
                # text = SpeechToText(file, language)
 
                # Use this to predict results
                # app_utils.DebugCommand("Text Recognised is ", text)
                translated = text
                translated = TranslateToEnglish(text, language)
                app_utils.DebugCommand("Translated to English Is ", translated)
                
                ModelColdStarter.join()
                prediction = PredictResults(translated)

                print(prediction[0])    
                f.write(file + ", " + prediction[0] + "\n")
                
