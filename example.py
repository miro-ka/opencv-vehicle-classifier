import cv2

# Load input image
image = cv2.imread('dataset/vehicle_detection_1.jpg')
# Convert image to gray to speed up classification
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Load classifier
classifier = cv2.CascadeClassifier('classifier/cascade.xml')
# Send image to classification. If the classifier could detect vehicles it will return
# list of rectangles
rects = classifier.detectMultiScale(gray,
        scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE)
# Draw a recatngle for every recatngle
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
# Show the final image
cv2.imshow("Vehicles", image)
cv2.waitKey(0)
