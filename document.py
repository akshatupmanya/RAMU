import cv2 as cv
import pytesseract



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
    #print(ocr_pipeline.recognize([img_data]))

    cv.imshow('frame',new_frame)

    key=cv.waitKey(1)
    if key==ord('q'):
        break

    if key==ord('c') and stability:
        text=pytesseract.image_to_string(new_frame)
        print(text)
        #language=detect(text)
        #print(language)

cam.release()
cv.destroyAllWindows()