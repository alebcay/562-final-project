from imutils import face_utils
import csv
import dlib
import cv2
import os

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

models = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
datapath = "C:/Users/caleb/COMP 562/Final Project/FERG_DB_orig"

anger_paths = [f'{datapath}/{model}/{model}_anger' for model in models]
disgust_paths = [f'{datapath}/{model}/{model}_disgust' for model in models]
fear_paths = [f'{datapath}/{model}/{model}_fear' for model in models]
joy_paths = [f'{datapath}/{model}/{model}_joy' for model in models]
neutral_paths = [f'{datapath}/{model}/{model}_neutral' for model in models]
sadness_paths = [f'{datapath}/{model}/{model}_sadness' for model in models]
surprise_paths = [f'{datapath}/{model}/{model}_surprise' for model in models]

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def getLandmarksForEmotion(paths, label):
	with open('cartoon_landmarks.csv', 'a+') as landmarkCsv:
		landmarkWriter = csv.writer(landmarkCsv)
		with open('cartoon_labels.csv', 'a+') as labelCsv:
			labelWriter = csv.writer(labelCsv)
			for path in paths:
				for file in absoluteFilePaths(path):
					print(file)
					image = cv2.imread(file)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					rects = detector(gray, 0)

					# For each detected face, find the landmark.
					for (i, rect) in enumerate(rects):
						# Make the prediction and transfom it to numpy array
						shape = predictor(gray, rect)
						shape = face_utils.shape_to_np(shape)

						flatshape = list(map(lambda i: image.shape[1] * i[0] + i[1], shape))
						landmarkWriter.writerow(flatshape)
						labelWriter.writerow([label])
		labelCsv.close()
	landmarkCsv.close()

getLandmarksForEmotion(anger_paths, 1)
getLandmarksForEmotion(disgust_paths, 2)
getLandmarksForEmotion(fear_paths, 3)
getLandmarksForEmotion(joy_paths, 4)
getLandmarksForEmotion(neutral_paths, 5)
getLandmarksForEmotion(sadness_paths, 6)
getLandmarksForEmotion(surprise_paths, 7)