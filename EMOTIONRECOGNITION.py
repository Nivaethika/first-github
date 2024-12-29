import cv2
from facial_emotion_recognition import EmotionRecognition

er=EmotionRecognition(device='cpu')
cam=cv2.VideoCapture(0)

while True:
    sucess,frames=cam.read()
    frames=er.recognise_emotion(frames,return_type='BGR')
    cv2.imshow("frame",frames)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
