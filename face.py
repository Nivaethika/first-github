import cv2
haar=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(grayimg,1.3,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,5),9)
    cv2.imshow("FACE DETECTION",img)
    key=cv2.waitKey(10)
    print(key)
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()
