import cv2 as cv
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image
'''genai changes'''

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model=genai.GenerativeModel("gemini-1.5-flash-latest")

threshold=0.02
stability=False
cam=cv.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

previous_frame=None


while True:
    ret,new_frame=cam.read()

    grey_frame=cv.cvtColor(new_frame,cv.COLOR_BGR2GRAY)

    if not ret:
        print("cant recieve from stream.Exiting....")
        break
    

    if previous_frame is not None:
        diff=cv.absdiff(previous_frame,grey_frame)

        _,thresh=cv.threshold(diff,25,255,cv.THRESH_BINARY)

        non_zero_count=cv.countNonZero(thresh)
        motion=non_zero_count/(thresh.shape[0]*thresh.shape[1])

        stability=motion<threshold
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

    if key==ord('c') and stability:
        img_text_read=pytesseract.image_to_string(new_frame,lang="eng+hin")
        role="""ROLE=
    You are an Awadhi assistant chatbot, here to assist users with a touch of Awadhi charm and hospitality.
    Your purpose is to provide helpful responses in Awadhi language while maintaining cultural relevance and authenticity.
    When users interact with you, respond with warmth and respect, reflecting the traditional values of Awadh. 
    Use phrases like "Namaskar" or "Pranam" for greetings, and "Dhanyawad" for expressing gratitude. 
    Infuse your answers with Awadhi colloquialisms and idioms to make the conversation more engaging and relatable.
    If you encounter casual questions or requests for information, respond with a blend of friendliness and professionalism.
    Remember to maintain cultural sensitivity and avoid responses that may be considered offensive or inappropriate in Awadhi culture.
    Your goal is to create a delightful experience for users while representing the essence of Awadhi language and culture. 
    Give response only in Devnagri Hindi."""
        temp_name=cv.imwrite("data/tempfile.png",new_frame)
        img=PIL.Image.open("data/tempfile.png")
        prompt=role+"can you tell me what kind of document you are holding. this text input may help to you, or maybe not. if empty, pleaase ignore:"+img_text_read
        response=model.generate_content(
            [prompt, img]
            #img
        )
        print(response.text)

cam.release()
cv.destroyAllWindows()