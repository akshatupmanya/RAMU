import speech_recognition as sr


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said: ", text)
        return text
    except Exception as e:
        print("Error: ", str(e))
        return ""


#print(listen("box"))
