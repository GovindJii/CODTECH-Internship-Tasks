import speech_recognition as sr
import os

def transcribe_audio_file(audio_path):
    # 1. Initialize the recognizer
    recognizer = sr.Recognizer()

    # 2. Load the audio file
    if not os.path.exists(audio_path):
        return "Error: File not found."

    print(f"--- Processing {audio_path} ---")
    
    with sr.AudioFile(audio_path) as source:
        # Record the audio data from the file
        audio_data = recognizer.record(source)
        
        try:
            print("Recognizing via Google Web Speech API...")
            # 3. Perform the transcription
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Error: Google could not understand the audio."
        except sr.RequestError as e:
            return f"Error: Could not request results; {e}"

# --- RUN THE TOOL ---
if __name__ == "__main__":
    # Make sure you have a file named 'sample.wav' in the same folder
    # or change this path to your file.
    filename = r"C:\Users\SanjuG\Videos\harvard.wav" 
    
    # Create a dummy file if it doesn't exist for demonstration
    if not os.path.exists(filename):
        print(f"Please place a .wav file named '{filename}' in this folder to test.")
    else:
        result = transcribe_audio_file(filename)
        print("\nTRANSCRIPTION:")
        print("="*40)
        print(result)
        print("="*40)