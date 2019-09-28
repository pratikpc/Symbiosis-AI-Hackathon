import os
import app_utils

app_utils.create_fullpath_if_not_exists("tmp")

# Command to call on command line to start translation
def GetCommand(language):
    return "python3 ./models/translator/translate.py -model ./models/translator/" + language + ".pt  -src ./tmp/input_translate.txt -output ./tmp/output_translated.txt -replace_unk"

def merged_text(text):
    text = text.split(' ')
    text_res = []
    for i in range(0, len(text),10):
        text_cur = text[i: i+9]
        text_cur = " ".join(text_cur)
        text_res.append(text_cur)
    text_res = "\n".join(text_res)
    return text_res

def TranslateToEnglish(text, source):
    # This translates to English
    if (source == "English"):
        return text
    text = merged_text(text)
    with open("tmp/input_translate.txt", "w") as inner:
        inner.write(text)
    os.system(GetCommand(source))
    with open("tmp/output_translated.txt", "r") as outer:
        text = outer.readlines()
    text = "".join(text)
    return text