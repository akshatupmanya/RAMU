import cv2 as cv
import pytesseract
import speech_recognition as sr
from IPython.display import Audio
from ttsmms import TTS

class mainLoader:
    def __init__(self) -> None:
        self.tts=TTS("data/awa")
        self.recognizer = sr.Recognizer()
    
    
    #for document scanning
    def document_loader(self):
        cam=cv.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Camera Not Open")
        previous_frame=None
        # our video loop
        while True:
            ret,new_frame=cam.read()
            grey_frame=cv.cvtColor(new_frame,cv.COLOR_BGR2GRAY)
            if not ret:
                raise Exception("cant recieve from stream.Exiting....")
            
            if previous_frame is not None:
                diff=cv.absdiff(previous_frame,grey_frame)
                _,thresh=cv.threshold(diff,25,255,cv.THRESH_BINARY)
                non_zero_count=cv.countNonZero(thresh)
                motion=non_zero_count/(thresh.shape[0]*thresh.shape[1])
                #our threshhold is 0.02
                stability=motion<0.02
                previous_frame=grey_frame.copy()
            else:
                previous_frame=grey_frame.copy()

            if stability:
                cv.putText(new_frame, "Stabilized", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            else:
                cv.putText(new_frame, "Chaos", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow('frame',new_frame)
            key=cv.waitKey(1)

            if key==ord('q'):
                break
            #our function to get document data
            if key==ord('c') and stability:
                text=pytesseract.image_to_string(new_frame)
                print(text)

        cam.release()
        cv.destroyAllWindows()

    #for speech recognition
    def speech_totext(self,box):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)

        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print("You said: ", text)
            return text
        except Exception as e:
            print("Error: ", str(e))
            return ""
        
    def text_to_speach(self,DataInput:str):
        wav=self.tts.synthesis(DataInput)
        Audio_file=Audio(wav["x"], rate=wav["sampling_rate"])
        return Audio_file
