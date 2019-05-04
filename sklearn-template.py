import sklearn.ensemble as skle
import sklearn.gaussian_process as sklgp
import sklearn.naive_bayes as sklnb
import sklearn.neural_network as sklnn
import sklearn.tree as sklt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def compareLabels(test_labels, model_labels):
	"""
	test_labels: Labels from test data.
	model_labels: Labels produces by model.
	"""
	test_labels=np.array(test_labels).reshape(1,-1)
	model_labels=np.array(model_labels).reshape(1,-1)
	if(np.size(test_labels) != np.size(model_labels)):
		print("yo jank is jank")
	else:
		print((np.size(test_labels)-np.count_nonzero(test_labels-model_labels))/np.size(test_labels))
	print()

TRAIN_PARAMETERS_CSV = "real_landmarks.csv"
TRAIN_FEATURES_CSV = "real_labels.csv"
TEST_PARAMETERS_CSV = "cartoon_landmarks.csv"
TEST_FEATURES_CSV = "cartoon_labels.csv"
print("Training on Real, Testing on Cartoon")

"""
TEST_PARAMETERS_CSV = "real_landmarks.csv"
TEST_FEATURES_CSV = "real_labels.csv"
TRAIN_PARAMETERS_CSV = "cartoon_landmarks.csv"
TRAIN_FEATURES_CSV = "cartoon_labels.csv"
print("Training on Cartoon, Testing on Real")
"""

x_train = np.array(pd.read_csv(TRAIN_PARAMETERS_CSV, header=None))
y_train = np.array(pd.read_csv(TRAIN_FEATURES_CSV, header=None)).ravel()
x_test = np.array(pd.read_csv(TEST_PARAMETERS_CSV, header=None))
y_test = np.array(pd.read_csv(TEST_FEATURES_CSV, header=None)).ravel()

NUM_TREES = 200

print("Normal Data:")
print("Random Forest with {} Trees".format(NUM_TREES))
rfc = skle.RandomForestClassifier(NUM_TREES)
rfc.fit(x_train, y_train)
compareLabels(rfc.predict(x_test),y_test)

print("Ada Boost")
abc = skle.AdaBoostClassifier()
abc.fit(x_train, y_train)
compareLabels(abc.predict(x_test),y_test)

print("Bagging")
bc = skle.BaggingClassifier()
bc.fit(x_train, y_train)
compareLabels(bc.predict(x_test),y_test)

print("Extra Trees")
etc = skle.ExtraTreesClassifier()
etc.fit(x_train, y_train)
compareLabels(etc.predict(x_test),y_test)

print("Gradient Boosting")
gbc = skle.GradientBoostingClassifier()
gbc.fit(x_train, y_train)
compareLabels(gbc.predict(x_test),y_test)

print("MLPClassifier")
mlpc = sklnn.MLPClassifier()
mlpc.fit(x_train, y_train)
compareLabels(mlpc.predict(x_test), y_test)

print("Decision Tree")
dtc = sklt.DecisionTreeClassifier()
dtc.fit(x_train, y_train)
compareLabels(dtc.predict(x_test), y_test)

print("Bernoulli NB")
bnb = sklnb.BernoulliNB()
bnb.fit(x_train, y_train)
compareLabels(bnb.predict(x_test), y_test)

print("Gaussian NB")
gnb = sklnb.GaussianNB()
gnb.fit(x_train, y_train)
compareLabels(gnb.predict(x_test), y_test)

print("Multinomial NB")
mnb = sklnb.MultinomialNB()
mnb.fit(x_train, y_train)
compareLabels(mnb.predict(x_test), y_test)

print("Gaussian Process Classifier")
gpc = sklgp.GaussianProcessClassifier()
gpc.fit(x_train, y_train)
compareLabels(gpc.predict(x_test), y_test)

"""
x_train = np.flip(x_train)
y_train = np.flip(y_train)
x_test = np.flip(x_test)
y_test = np.flip(y_test)

print("Flipped Data:")
print("Random Forest with {} Tree".format(NUM_TREES))
rfc_f = skle.RandomForestClassifier(NUM_TREES)
rfc_f.fit(x_train, y_train)
compareLabels(rfc_f.predict(x_test),y_test)

print("Ada Boost")
abc_f = skle.AdaBoostClassifier()
abc_f.fit(x_train, y_train)
compareLabels(abc_f.predict(x_test),y_test)

print("Bagging")
bc_f = skle.BaggingClassifier()
bc_f.fit(x_train, y_train)
compareLabels(bc_f.predict(x_test),y_test)

print("Extra Trees")
etc_f = skle.ExtraTreesClassifier()
etc_f.fit(x_train, y_train)
compareLabels(etc_f.predict(x_test),y_test)

print("Gradient Boosting")
gbc_f = skle.GradientBoostingClassifier()
gbc_f.fit(x_train, y_train)
compareLabels(gbc_f.predict(x_test),y_test)
"""