import cv2

camera = cv2.VideoCapture(0)
while True:
    (grapped, frame) = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Face", gray)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()