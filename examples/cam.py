import cv2


cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('../classifier/cascade.xml')

while True:
    grab, frame = cam.read()
    # Scale the frame for faster classification
    height, width = frame.shape[:2]
    scaleRate = 300/width
    frame = cv2.resize(frame, (0,0), fx=scaleRate, fy=scaleRate)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Run classification
    rects = classifier.detectMultiScale(gray,
        scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20))
    frameClone = frame.copy()

    for (x, y, w, h) in rects:
        cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Vehicle detection example", frameClone)

    if cv2.waitKey(20) == 27: # exit on ESC
        break

cam.release()
cv2.destroyAllWindows()
