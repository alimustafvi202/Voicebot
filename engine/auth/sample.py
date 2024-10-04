import cv2
import os

# Create necessary directories
if not os.path.exists('engine/auth/samples/'):
    os.makedirs('engine/auth/samples/')

# Create a video capture object
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

cam.set(3, 640)  # Set video Frame Width
cam.set(4, 480)  # Set video Frame Height

# Load Haar Cascade classifier
detector = cv2.CascadeClassifier('engine/auth/haarcascade_frontalface_default.xml')

face_id = input("Enter a Numeric user ID here: ")
print("Taking samples, look at camera... ")
count = 0  # Initializing sampling face count

while True:
    ret, img = cam.read()  # Read the frames
    if not ret:
        print("Error: Could not read image.")
        break

    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector.detectMultiScale(converted_image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle
        count += 1

        # Save images to dataset folder
        cv2.imwrite(f"engine/auth/samples/face.{face_id}.{count}.jpg", converted_image[y:y + h, x:x + w])

        cv2.imshow('image', img)  # Display the image

    k = cv2.waitKey(100) & 0xff  # Wait for a key
    if k == 27:  # Press 'ESC' to stop
        break
    elif count >= 100:  # Take 100 samples
        break

print("Samples taken, now closing the program...")
cam.release()
cv2.destroyAllWindows()
