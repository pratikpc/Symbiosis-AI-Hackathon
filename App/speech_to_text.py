# Import necessary libraries 
from pydub import AudioSegment 
import speech_recognition as sr 
import os
import shutil

def SpeechToText(fileName, language = 'en-US'):
    os.makedirs("audio_chunks", exist_ok=True)
    # Input audio file to be sliced 
    audio = AudioSegment.from_wav(fileName) 

    ''' 
    Step #1 - Slicing the audio file into smaller chunks. 
    '''
    # Length of the audiofile in milliseconds 
    n = len(audio) 

    # Variable to count the number of sliced chunks 
    counter = 1

    # Text file to write the recognized audio 

    # Interval length at which to slice the audio file. 
    # If length is 22 seconds, and interval is 5 seconds, 
    # The chunks created will be: 
    # chunk1 : 0 - 5 seconds 
    # chunk2 : 5 - 10 seconds 
    # chunk3 : 10 - 15 seconds 
    # chunk4 : 15 - 20 seconds 
    # chunk5 : 20 - 22 seconds 
    interval = 10 * 1000

    # Length of audio to overlap.  
    # If length is 22 seconds, and interval is 5 seconds, 
    # With overlap as 1.5 seconds, 
    # The chunks created will be: 
    # chunk1 : 0 - 5 seconds 
    # chunk2 : 3.5 - 8.5 seconds 
    # chunk3 : 7 - 12 seconds 
    # chunk4 : 10.5 - 15.5 seconds 
    # chunk5 : 14 - 19.5 seconds 
    # chunk6 : 18 - 22 seconds 
    overlap =  3 * 1000

    # Initialize start and end seconds to 0 
    start = 0
    end = 0

    # Flag to keep track of end of file. 
    # When audio reaches its end, flag is set to 1 and we break 

    flag = 0

    # Iterate from 0 to end of the file, 
    # with increment = interval 

    recognised_text = ""

    for i in range(0, 2 * n, interval): 
        
        # During first iteration, 
        # start is 0, end is the interval 
        if i == 0: 
            start = 0
            end = interval 

        # All other iterations, 
        # start is the previous end - overlap 
        # end becomes end + interval 
        else: 
            start = end - overlap 
            end = start + interval  

        # When end becomes greater than the file length, 
        # end is set to the file length 
        # flag is set to 1 to indicate break. 
        if end >= n: 
            end = n 
            flag = 1

        # Storing audio file from the defined start to end 
        chunk = audio[start:end] 

        # Filename / Path to store the sliced audio 
        # We store the data here temporarily
        # Then we will delete hte data
        filename = 'audio_chunks/chunk'+str(counter)+'.wav'

        # Store the sliced audio file to the defined path 
        chunk.export(filename, format ="wav") 
        # # Print information about the current chunk 
        # print("Processing chunk "+str(counter)+". Start = "
        #                     +str(start)+" end = "+str(end)) 

        # Increment counter for the next chunk 
        counter = counter + 1

        AUDIO_FILE = filename 

        # Initialize the recognizer 
        r = sr.Recognizer() 
        
        
        ## recognizer properties refer https://pypi.org/project/SpeechRecognition
        #r.pause_threshold = 0.8
        #r.energy_threshold = 500
        #r.dynamic_energy_threshold = True
        #r.dynamic_energy_adjustment_damping = 0.15
        #r.dynamic_energy_adjustment_ratio = 1.5

        # Traverse the audio file and listen to the audio 
        with sr.AudioFile(AUDIO_FILE) as source:
        
            audio_listened = r.listen(source) 

        # Try to recognize the listened audio 
        # And catch expections. 
        try:     
            #translating the sliced audio into english text and saving it into a file
            rec_eng = r.recognize_google(audio_listened, language = language) 
            recognised_text = recognised_text + " " + rec_eng
        
        # If google could not understand the audio 
        except sr.UnknownValueError: 
            pass

        # If the results cannot be requested from Google. 
        # Probably an internet connection error. 
        except sr.RequestError as e: 
            pass

        # Check for flag. 
        # If flag is 1, end of the whole audio reached. 
        # Close the file and break. 
        if flag == 1: 
            break
    shutil.rmtree("audio/chunks")

    return recognised_text