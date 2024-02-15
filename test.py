import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    (grapped, frame) = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in rects:
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY+ fH), (0, 255, 0), 2)
    cv2.putText(frame, "Aamin here" , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 1)
    
    cv2.imshow("Face", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

# frame  = cv2.imread("./ananya.png")
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
# for (fX, fY, fW, fH) in rects:
#     cv2.rectangle(frame, (fX, fY), (fX + fW, fY+ fH), (0, 255, 0), 2)
# cv2.putText(frame, "Ananya here" , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 1)

# cv2.imshow("Face", frame)
# cv2.waitKey(0)
# cv2.destroyAWllindows()
