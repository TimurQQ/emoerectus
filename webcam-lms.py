import cv2
import numpy as np
import dlib
import glob
import os
import random
from PIL import Image
import math
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
emotions = ["happy", "anger", "fear"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clf_weights_filename = 'finalized_model.sav'
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
clf = SVC(kernel='linear', probability=True, tol=1e-5)
base_dir = os.path.dirname(__file__)


def get_files(emotion):
    files = glob.glob(os.path.join(base_dir, "dataset",
                                   f'{emotion}_set2', '*'))
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    testing = files[-int(len(files)*0.2):]
    return training, testing


def get_landmarks(image):
    detections = detector(image, 1)
    landmarks_vectorised = "error"
    xlist = "error"
    ylist = "error"
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist, ylist = [], []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean, ymean = np.mean(xlist), np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y in zip(xcentral, ycentral):
            dist = np.linalg.norm(np.array((x, y)))
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*np.pi))
    return np.array(xlist), np.array(ylist), landmarks_vectorised


def plt_show_detection(xarr, yarr,clahe_image):
    f, (ax1, ax2) = plt.subplots(1, 2)
    xmean, ymean = np.mean(xarr), np.mean(yarr)
    ax1.imshow(clahe_image)
    ax1.scatter(xarr, yarr, color = 'red', s = 4)
    ax1.scatter(xmean, ymean, color = 'blue', s = 9)
    PIL_img = Image.new('RGB', clahe_image.shape, (255, 255, 255))
    open_cv_image = np.array(PIL_img)[:, :, ::-1].copy()
    ax2.imshow(open_cv_image)
    ax2.scatter(xarr, yarr, color = 'red', s = 4)
    ax2.scatter(xmean, ymean, color = 'blue', s = 9)
    plt.show()

def make_sets():
    training_data, training_labels = [], []
    testing_data, testing_labels = [], []
    for emotion in emotions:
        print(f" working on {emotion}")
        training, testing = get_files(emotion)
        print("\nTrainingSet\n")
        for img_path in training:
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            xarr, yarr, landmarks_vectorised = get_landmarks(clahe_image)
            plt_show_detection(xarr, yarr, clahe_image)
            if landmarks_vectorised == "error":
                print("no face detected on this one")
            else:
                training_data.append(landmarks_vectorised)
                training_labels.append(emotions.index(emotion))
        print("\nTestingSet:\n")
        for item in testing:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            xarr, yarr, landmarks_vectorised = get_landmarks(clahe_image)
            plt_show_detection(xarr, yarr, clahe_image)
            if landmarks_vectorised == "error":
                print("no face detected on this one")
            else:
                testing_data.append(landmarks_vectorised)
                testing_labels.append(emotions.index(emotion))
    return training_data, training_labels, testing_data, testing_labels


def train_clf():
    print("Making sets")
    training_data, training_labels, testing_data, testing_labels = make_sets()
    npar_train = np.array(training_data)
    print("training SVM linear")
    print(npar_train)
    clf.fit(npar_train, training_labels)
    print("getting accuracies")
    npar_test = np.array(testing_data)
    pred_lin = clf.score(npar_test, testing_labels)
    print ("linear: ", pred_lin)
    pickle.dump(clf, open(clf_weights_filename, 'wb'))


def webcam_detect():
    video_capture = cv2.VideoCapture(0)
    while(video_capture.isOpened()):
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        xarr, yarr, landmarks_vectorised = get_landmarks(clahe_image)
        if landmarks_vectorised != "error":
            pred_prob = clf.predict_proba([landmarks_vectorised])
            print(pred_prob)
            print(emotions[pred_prob[0].argmax()])
        if list(xarr) != 'error' and list(yarr) != 'error':
            for x, y in zip(xarr, yarr):
                cv2.circle(frame, (int(x),int(y)), 2, (0, 0, 255))
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("q pressed")
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    train_clf()
    clf = pickle.load(open(clf_weights_filename, 'rb'))
    webcam_detect()
