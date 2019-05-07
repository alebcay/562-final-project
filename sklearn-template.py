import sklearn.ensemble as skle
import sklearn.gaussian_process as sklgp
import sklearn.naive_bayes as sklnb
import sklearn.neural_network as sklnn
import sklearn.tree as sklt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

VALID_FILE_TYPES = ["real", "cartoon", "combined"]
OUTPUT_FILE = "sklearn_output.csv"
NUM_TREES = 200

def compareLabels(test_labels, model_labels):
	test_labels=np.array(test_labels).reshape(1,-1)
	model_labels=np.array(model_labels).reshape(1,-1)
	if(np.size(test_labels) != np.size(model_labels)):
		print("Mismatched length of test_labels (length {}) and model_labels (length {}) for model {}"
			.format(np.size(test_labels), np.size(model_labels), model_name))
		return None
	else:
		return str((np.size(test_labels)-np.count_nonzero(test_labels-model_labels))/np.size(test_labels))

pairs_to_run = []
if(len(sys.argv)!=1):
	if(len(sys.argv)%2 != 1):
		print("Invalid parameters. Number of parameters must be even.")
	else:
		for i in range(1, len(sys.argv), 2):
			if sys.argv[i] in VALID_FILE_TYPES and sys.argv[i+1] in VALID_FILE_TYPES:
				pairs_to_run.append([sys.argv[i],sys.argv[i+1]])
			else:
				print("Invalid parameter. Valid types are {}".format(VALID_FILE_TYPES))

if len(pairs_to_run) == 0: #No command-line arguments
	for i in VALID_FILE_TYPES:
		for j in VALID_FILE_TYPES:
			pairs_to_run.append([i,j])

output = {}
output_cols = ["Model"]

for pair in pairs_to_run:
	print("{} => {}".format(pair[0], pair[1]))
	output_cols.append("{} => {}".format(pair[0], pair[1]))
	train_parameters_csv = "{}_landmarks_train.csv".format(pair[0])
	train_features_csv = "{}_labels_train.csv".format(pair[0])
	test_parameters_csv = "{}_landmarks_test.csv".format(pair[1])
	test_features_csv = "{}_labels_test.csv".format(pair[1])
	print("Training on {}, Testing on {}".format(pair[0], pair[1]))

	x_train = np.array(pd.read_csv(train_parameters_csv, header=None))
	y_train = np.array(pd.read_csv(train_features_csv, header=None)).ravel()
	x_test = np.array(pd.read_csv(test_parameters_csv, header=None))
	y_test = np.array(pd.read_csv(test_features_csv, header=None)).ravel()

	print("Random Forest with {} Trees".format(NUM_TREES))
	rfc = skle.RandomForestClassifier(NUM_TREES)
	rfc.fit(x_train, y_train)
	output.setdefault("Random Forest with {} Trees".format(NUM_TREES), []).append(compareLabels(rfc.predict(x_test),y_test))
	
	print("Ada Boost")
	abc = skle.AdaBoostClassifier()
	abc.fit(x_train, y_train)
	output.setdefault("Ada Boost", []).append(compareLabels(abc.predict(x_test),y_test))

	print("Bagging")
	bc = skle.BaggingClassifier()
	bc.fit(x_train, y_train)
	output.setdefault("Bagging", []).append(compareLabels(bc.predict(x_test),y_test))

	print("Extra Trees")
	etc = skle.ExtraTreesClassifier()
	etc.fit(x_train, y_train)
	output.setdefault("Extra Trees", []).append(compareLabels(etc.predict(x_test),y_test))

	print("Gradient Boosting")
	gbc = skle.GradientBoostingClassifier()
	gbc.fit(x_train, y_train)
	output.setdefault("Gradient Boosting", []).append(compareLabels(gbc.predict(x_test),y_test))

	print("MLPClassifier")
	mlpc = sklnn.MLPClassifier()
	mlpc.fit(x_train, y_train)
	output.setdefault("MLPClassifier", []).append(compareLabels(mlpc.predict(x_test), y_test))

	print("Decision Tree")
	dtc = sklt.DecisionTreeClassifier()
	dtc.fit(x_train, y_train)
	output.setdefault("Decision Tree", []).append(compareLabels(dtc.predict(x_test), y_test))

	print("Bernoulli NB")
	bnb = sklnb.BernoulliNB()
	bnb.fit(x_train, y_train)
	output.setdefault("Bernoulli NB", []).append(compareLabels(bnb.predict(x_test), y_test))

	print("Gaussian NB")
	gnb = sklnb.GaussianNB()
	gnb.fit(x_train, y_train)
	output.setdefault("Gaussian NB", []).append(compareLabels(gnb.predict(x_test), y_test))

	print("Multinomial NB")
	mnb = sklnb.MultinomialNB()
	mnb.fit(x_train, y_train)
	output.setdefault("Multinomial NB", []).append(compareLabels(mnb.predict(x_test), y_test))

file = open(OUTPUT_FILE, "w+")
for header in output_cols:
	file.write(header+",")
file.write("\n")
for row in output.keys():
	file.write(row+",")
	for val in output[row]:
		file.write(val+",")
	file.write("\n")
file.close()