import cv2
import os

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

base_dir = os.path.dirname(__file__)

if os.path.exists(os.path.join(base_dir, 'dataset')):
    os.system(f"rm -r {os.path.join(base_dir, 'dataset')}")
os.system(f"mkdir {os.path.join(base_dir, 'dataset')}")

def detect_faces(emotion):
    files_dir = os.path.join(base_dir,'sorted_set',f'{emotion}') #Get list of all images with emotion
    data_dir = os.path.join(base_dir, 'dataset', f'{emotion}')
    
    if os.path.exists(data_dir):
        os.system(f"rm -r {data_dir}")
    os.system(f"mkdir {data_dir}")
    
    filenumber = 0
    for f in os.listdir(files_dir):
        img_dir = os.path.join(files_dir, f)
        frame = cv2.imread(img_dir) #Open image

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""

        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print (f"face found in file: {f}")
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            
            #cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            print(os.path.join(data_dir, f'{filenumber}.png'))
            cv2.imwrite(os.path.join(data_dir, f'{filenumber}.png'), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion) #Call functiona
