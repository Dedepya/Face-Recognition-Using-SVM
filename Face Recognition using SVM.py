import os
from tkinter import *
import tkinter.font as font
import tkinter.simpledialog as simpledialog
from PIL import Image, ImageTk
from tkinter import messagebox as mbox

import cv2
import face_recognition
from numpy import info
from sklearn import svm


def test_svm():
    
    # Training the SVC classifier

    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    train_dir = os.listdir('train/')

    for person in train_dir:
        pix = os.listdir("train/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                "train/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img +" was skipped and can't be used for training")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file('test/test.jpg')

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    num = len(face_locations)
    print("Number of faces detected: ", num)

    # Predict all the faces in the test image using the trained classifier
    list_names = []
    print("Found:")
    for i in range(num):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        print(name)
        list_names.append(*name)
    display_name(list_names)


def test_img_capture():

    window.destroy()

    img_counter = 0
   
    if(True):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Face Recognition")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Face Recognition", frame)

            k = cv2.waitKey(1)
            if k % 256 == 32:
                # SPACE pressed
                img_name = "test.jpg"
                cv2.imwrite(os.path.join('test', img_name), frame)
                print("{} written!".format(img_name))
                print("Closing now")
                img_counter += 1
                break
    cam.release()
    cv2.destroyAllWindows()
    test_svm()

def train_img_capture():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Face Training")
    img_counter = 0

    file_name = ''
    file_name= simpledialog.askstring(title="Face Recognition",prompt="What's your Name?:")
    window = Tk()
    window.withdraw()
    
    
    os.mkdir("train/"+file_name)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Face Training", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = file_name+"_{}.jpg".format(img_counter)
            cv2.imwrite(os.path.join("train/"+file_name, img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
    window.destroy()


def display_name(list_name): 
    window=Tk()
    label = Label(window, text="Faces Recognized")
    listbox = Listbox(window, width=50)
    label.pack()
    listbox.pack(fill=BOTH, expand=1)  # adds listbox to window
    for row in list_name:
        listbox.insert(END, row)   # one line for loop
    window.mainloop()


window=Tk()
window.config(width=300, height=300,padx=20,pady=50)
label = Label(
    window, text='WELCOME TO FACE RECOGNITION. SELECT AN OPTION BELOW:\n',font=font.Font(size=16))
label.pack()
button = Button(window, text="Train",command=train_img_capture,width=20,bg="red",fg="white",pady=10)
button['font']=font.Font(size=16)
button.pack()
label = Label(window, text='\n')
label.pack()
button = Button(window, text="Test", command=test_img_capture,width=20,bg="#0052cc",fg="white",pady=10)
button['font']=font.Font(size=16)
button.pack()
label=Label(window,text="\nInstructions\n1).In Train Mode, enter your name and then press SPACEBAR to capture images. Hit ESC when done.\n2).In Test Mode, press SPACEBAR to capture image and display detected face",font=font.Font(size=14))
label.pack()
window.mainloop()




