#Import required modules
import cv2
import numpy as np
import dlib
import glob
import os
import random
import math
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

emotions = ["anger", "happy"]#, "contempt", "disgust", "fear", "neutral", "sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clf_weights_filename = 'finalized_model.sav'

#Set up some required objects
detector = dlib.get_frontal_face_detector() #Face detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path) #Landmark identifier. Set the filename to whatever you named the downloaded file

clf = SVC(kernel='linear', probability=True, tol=1e-5)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

base_dir = os.path.dirname(__file__)

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(os.path.join(base_dir,"dataset",f'{emotion}_set2','*'))
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    testing = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, testing

def get_landmarks(image):
    detections = detector(image, 1)
    
    landmarks_vectorised = "error"
    lm_types = "error"
    #landmarks = "error"
    for k, d in enumerate(detections): #For each detected face
        shape = predictor(image, d) #Get coordinates
        #landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    
        xlist = []
        ylist = []
        
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]
        
        landmarks_vectorised = []
        
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        lm_types = {
        'lower_face': (shape.parts()[0: 17], (255, 0, 0)),
        'eyebrow1': (shape.parts()[17: 22], (0, 255, 0)),
        'eyebrow2': (shape.parts()[22: 27], (0, 255, 0)),
        'nose': (shape.parts()[27: 31], (255, 0, 255)),
        'nostril': (shape.parts()[31: 36], (255, 255, 0)),
        'eye1': (shape.parts()[36: 42], (0, 0, 255)),
        'eye2': (shape.parts()[42: 48], (0, 0, 255)),
        'lips': (shape.parts()[48: 60], (0, 255, 0)),
        'teeth': (shape.parts()[60: 68], (0, 0, 0))
        }
    return lm_types, landmarks_vectorised

def make_sets():
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    for emotion in emotions:
        print(f" working on {emotion}")
        training, testing = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        print("\nTrainingSet\n")
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            lm_types, landmarks_vectorised = get_landmarks(clahe_image)
            if lm_types != "error":
                for lm_unit in lm_types.values():
                    for lm in lm_unit[0]:
                        cv2.circle(clahe_image, (lm.x, lm.y), 2, lm_unit[1], -1) 
            plt.imshow(clahe_image)
            plt.show()
            
            if landmarks_vectorised == "error":
                print("no face detected on this one")
            else:
                training_data.append(landmarks_vectorised) #append image array to training data list
                training_labels.append(emotions.index(emotion))
                
        print("\nTestingSet:\n")
        for item in testing:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            
            lm_types, landmarks_vectorised = get_landmarks(clahe_image)
            if lm_types != "error":
                for lm_unit in lm_types.values():
                    for lm in lm_unit[0]:
                        cv2.circle(clahe_image, (lm.x, lm.y), 2, lm_unit[1], -1) 
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(clahe_image)
            print(landmarks_vectorised)
            ax2.plot()
            plt.show()
            plt.figure()
            
            if landmarks_vectorised == "error":
                print("no face detected on this one")
            else:
                testing_data.append(landmarks_vectorised)
                testing_labels.append(emotions.index(emotion))
    return training_data, training_labels, testing_data, testing_labels

def train_clf():
    #accur_lin = []
    for i in range(0,1):
        print(f"Making sets {i+1}") #Make sets by random sampling 80/20%
        training_data, training_labels, testing_data, testing_labels = make_sets()
        npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
        #npar_trainlabs = np.array(training_labels)
        print(f"training SVM linear {i+1}") #train SVM
        print(npar_train)
        clf.fit(npar_train, training_labels)
        print(f"getting accuracies {i+1}") #Use score() function to get accuracy
        npar_test = np.array(testing_data)
        pred_lin = clf.score(npar_test, testing_labels)
        print ("linear: ", pred_lin)
        #accur_lin.append(pred_lin) #Store accuracy in a list
        #print(f"Mean value lin svm: {np.mean(accur_lin)}") #FGet mean accuracy of the 10 runs
    pickle.dump(clf, open(clf_weights_filename, 'wb'))

def webcam_detect():
    video_capture = cv2.VideoCapture(0) #Webcam object
    while(video_capture.isOpened()):
    
        _, frame = video_capture.read()
        #frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        #cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        lm_types, landmarks_vectorised = get_landmarks(clahe_image)
        for lm_unit in lm_types.values():
            for lm in lm_unit[0]:
                cv2.circle(frame, (lm.x, lm.y), 2, lm_unit[1], -1) 
        
        cv2.imshow("image", frame) #Display the frame
    
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            print("q pressed")    
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_clf()
    
    clf = pickle.load(open(clf_weights_filename, 'rb'))
    
    webcam_detect()
