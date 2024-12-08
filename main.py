import cv2

def blur_face(img):
    (h, w) = img.shape[:2]
    dW = int(w / 3.0)
    dH = int(h / 3.0)
    if dW %2 == 0:
        dW -= 1
    if dH %2 == 0:
        dH -= 1
    return cv2.GaussianBlur(img,(dW, dH), 0)

capture = cv2.VideoCapture(8)
face_cascade = cv2.CascadeClassifier('haarcascade frontalface default.xml')

while True:
    ret, img = capture.read()

    faces = face_cascade.detectMultiScale(img, scaleFactor=2.0  , minNeighbors=5, minSize=(28, 20))

    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img[y:y+h, x:x+w] = blur_face(img[y:y+h, x:x+w])

    cv2.imshow('From Camera', img)

    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

capture.release()
cv2.destroyAllWindows