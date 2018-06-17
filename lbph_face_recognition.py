import numpy as np
import cv2
import os
import random

# Function to load training data
def load_data(dir):

    # Get list of directories. Each directory has photos of one person
    directories = [d for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))]

    # placeholders for images and corresponding labels
    labels = []
    images = []
    actual_labels = []

    print(len(directories))
    label_number = 0
    for d in directories:
        # Get path of current directory
        label_dir = os.path.join(dir, d)
        # Get names of all images in the current directory
        file_names = [os.path.join(label_dir, f)
                    for f in os.listdir(label_dir)
                    if f.endswith(".jpg")]
        # Read images
        for f in file_names:
            img = cv2.imread(f)
            # make sure the image is of the correct dimension
            #imgresized = cv2.resize(img, (250, 250))

            # Detect faces in the image
            face_cropped, rect = detect_face(img)

            # Ignore undetected faces
            if face_cropped is not None:
                # Add the image and corresponding label to our list
                images.append(face_cropped)
                labels.append(label_number)

        # Find number of pictures in the current directory
        num_of_pics= len(os.listdir(os.path.join(dir, d)))
        if num_of_pics > 1:
            print(d, num_of_pics)
        # Add new class
        actual_labels.append(d)
        # increment label number
        label_number += 1
        # Early Stop
        if label_number >501:
            print("breaking")
            break

    return labels, images, actual_labels

# Function to detect and extract faces from images
def detect_face(image):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #load OpenCV LBPH face detector (Local Binary Patterns Histograms)
    face_cascade = cv2.CascadeClassifier('/home/sattvik/computer_vision/LBPH/bin/opencv/data/lbpcascades/lbpcascade_frontalface.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face, extract the face area
    x, y, w, h = faces[0]

    #return only the face part of the image
    return gray_img[y:y+w, x:x+h], faces[0]

# Function to draw a rectange on a given image
def draw_rectangle(img, rect):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to add text on face box
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)    

# Function to predict faces
def predict(test_img, recognizer, labels):
    # Make a copy of the image so that the original does not get overwritten
    img = test_img.copy()
    # Detect faces in the test image
    face, rect = detect_face(img)

    # Make prediction
    predicted_label = recognizer.predict(face)

    # Find the name of the preson in the picture
    label_text = labels[predicted_label[0]]

    # Draw a rectange around the face
    draw_rectangle(img, rect)

    # Add the person's name
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img


# Main execution sequence
# Load and prepare data
training_labels, training_images, training_actual_labels = load_data('/home/sattvik/Labeled_faces_in_the_wild_dataset/truncated_dataset')
print("Data loaded and prepared")
print("Data dimensions:")
print(len(training_labels))
print(len(training_images))

# Sanity Check
print("Sanity Check: Print random face")
random_example = random.randint(0,(len(training_labels)-1))
print(training_labels[random_example])
cv2.imshow(training_actual_labels[training_labels[random_example]], training_images[random_example])
cv2.waitKey(0)
cv2.destroyAllWindows()

#Initialize our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Train the face recognizer
face_recognizer.train(training_images, np.array(training_labels))
print("Training Complete")

# load test data
test_dir = '/home/sattvik/Labeled_faces_in_the_wild_dataset/truncated_test_data'
test_image_path = 'Sattvik/me_test_001.jpg'
test_image = cv2.imread(os.path.join(test_dir, test_image_path))
cv2.imshow(test_image_path, test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prediction
print("Making prediction")
final_image = predict(test_image, face_recognizer, training_actual_labels)
cv2.imshow("Final test image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()