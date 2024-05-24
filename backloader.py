import cv2 as cv
import os
import pytesseract
import speech_recognition as sr
from IPython.display import Audio
from ttsmms import TTS
import wave
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import PIL
import time
import pyttsx3
from playsound import playsound
import io
from gtts import gTTS
import pygame
import os


def speak(text):
    engine = pyttsx3.init()
    # Initialize gTTS with the desired language and speed
    tts = gTTS(text=text, lang="hi")  # 'hi' for Hindi, 'en' for English

    # Create a BytesIO object to store the audio
    audio_stream = io.BytesIO()
    # Save the audio to the BytesIO object
    tts.write_to_fp(audio_stream)
    # Reset the BytesIO object's position to the beginning
    audio_stream.seek(0)
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio from the BytesIO object
    pygame.mixer.music.load(audio_stream)

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue
    engine.runAndWait()


class mainLoader:
    def __init__(self) -> None:
        self.tts = TTS("data/awa")
        self.recognizer = sr.Recognizer()
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.ROLE = """ROLE=
    You are an Awadhi assistant chatbot, here to assist users with a touch of Awadhi charm and hospitality.
    Your purpose is to provide helpful responses in Awadhi language while maintaining cultural relevance and authenticity.
    When users interact with you, respond with warmth and respect, reflecting the traditional values of Awadh. 
    Use phrases like "Namaskar" or "Pranam" for greetings, and "Dhanyawad" for expressing gratitude. 
    Infuse your answers with Awadhi colloquialisms and idioms to make the conversation more engaging and relatable.
    If you encounter casual questions or requests for information, respond with a blend of friendliness and professionalism.
    Remember to maintain cultural sensitivity and avoid responses that may be considered offensive or inappropriate in Awadhi culture.
    Your goal is to create a delightful experience for users while representing the essence of Awadhi language and culture. 
    Give response only in Devnagri Hindi."""

    def gemini_callfunc(self, prompt, degree: int, data=None):
        model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
        )
        if degree == 1:
            response = model.generate_content(self.ROLE + prompt)
        elif degree == 2:
            response = model.generate_content([prompt, data])
        else:
            raise Exception("DegreeError. The Input degree function is exceeded")
        speak(response.text)
        return response.text

    # for document scanning
    def document_loader(self):
        stability = False
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Camera Not Open")
        previous_frame = None
        # our video loop
        while True:
            ret, new_frame = cam.read()
            grey_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
            if not ret:
                raise Exception("cant recieve from stream.Exiting....")

            if previous_frame is not None:
                diff = cv.absdiff(previous_frame, grey_frame)
                _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
                non_zero_count = cv.countNonZero(thresh)
                motion = non_zero_count / (thresh.shape[0] * thresh.shape[1])
                # our threshhold is 0.02
                stability = motion < 0.02
                previous_frame = grey_frame.copy()
            else:
                previous_frame = grey_frame.copy()

            if stability:
                cv.putText(
                    new_frame,
                    "Stabilized",
                    (50, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            else:
                cv.putText(
                    new_frame,
                    "Chaos",
                    (50, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            cv.imshow("frame", new_frame)
            key = cv.waitKey(1)

            if key == ord("q"):
                cam.release()
                cv.destroyAllWindows()
                return None
            # our function to get document data
            if key == ord("c") and stability:
                img_text_read = pytesseract.image_to_string(new_frame)
                cv.imwrite("data/tempfile.png", new_frame)
                img = PIL.Image.open("data/tempfile.png")
                prompt = (
                    self.ROLE
                    + "can you tell me what kind of document you are holding. this text input may help to you, or maybe not. if empty, pleaase ignore:"
                    + img_text_read
                )
                pic_data = self.gemini_callfunc(prompt=prompt, data=img, degree=2)
                cam.release()
                cv.destroyAllWindows()
                return pic_data

    # for speech recognition
    def speech_totext(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print("You said: ", text)
            prompt = self.ROLE + text
            data_output = self.gemini_callfunc(prompt=prompt, degree=1)
            return data_output
        except Exception as e:
            raise Exception("Error: ", str(e))

    def text_to_speach(
        self,
        DataInput: str = """ठीक है, बताई देता हूँ। आप जानना चाहते हैं कि कौन हैं श्री राम?

श्री राम हिन्दू धर्म के प्रमुख देवताओं में से एक हैं। उन्हें मर्यादा पुरुषोत्तम के नाम से भी जाना जाता है। रामायण की कथा उन्हीं के जीवन पर आधारित है।

श्री राम को आदर्श पुत्र, आदर्श भाई, आदर्श पति और आदर्श राजा माना जाता है।  उन्होंने अपने जीवन में हर कर्तव्य को बखूबी निभाया।

कुछ और जानना चाहते हैं आप?""",
    ):
        wav = self.tts.synthesis(DataInput)
        wav_file = "output.wav"
        with wave.open(wav_file, "w") as f:
            f.setnchannels(2)
            f.setsampwidth(2)

            f.setframerate(wav["sampling_rate"])
            f.writeframes(np.array(wav["x"], dtype=np.int16).tobytes())
        Audio_file = Audio(
            wav["x"],
            rate=wav["sampling_rate"],
            autoplay=True,
        )

        with open(wav_file, "rb") as f:
            audio_bytes = f.read()
        pygame.mixer.init()
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
            # return Audio_file
        os.popen("start output.wav")
