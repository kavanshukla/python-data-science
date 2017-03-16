import numpy as np
import scipy 
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import metrics
import matplotlib.pyplot as plt

data = np.load('forest_data.npz')
label_training = np.array(data['label_training'])
data_training = scale(data['data_training'])
data_val = data['data_val']
label_val = data['label_val']

sampling_rates = np.array([0.2, 0.4, 0.6, 0.8])
forest_size = np.array([10, 20, 50])

oob_error_feature_sampling = []
oob_error_data_sampling = []

print "Experiment 1: "		
print "Out of Bag error using different feature sampling rates: "

for i in range(0,len(forest_size)):
	oob_error_feature_sampling = []
	for j in range(0,len(sampling_rates)):
		model = RandomForestClassifier(n_estimators = forest_size[i], max_features = sampling_rates[j], oob_score = True, bootstrap = True, max_depth = 100, class_weight = "balanced_subsample")
		model.fit(data_training, label_training)
		predictions = model.predict(data_val)
		oob_error = 1 - model.oob_score_
		oob_error_feature_sampling.append(oob_error)
		print("Forest Size: "+ str(forest_size[i])+ " Feature Sampling Rate: " + str(sampling_rates[j]) + " Out of Bag Error: " + str(oob_error))
	plt.plot(sampling_rates, oob_error_feature_sampling, label="Forest = "+str(forest_size[i]))

plt.xlabel("Feature Sampling Rates")
plt.ylabel("Out of Bag Error Rate")
plt.legend()
plt.show()

print("Experiment 2: ")		
print("Out of Bag error using different data sampling rates: ")

for i in range(0,len(forest_size)):
	oob_error_data_sampling = []
	for j in range(0,len(sampling_rates)):
		model = RandomForestClassifier(n_estimators = forest_size[i], oob_score = True, bootstrap = True, max_depth = 100, class_weight = "balanced_subsample")
		model.fit(data_training, label_training, sample_weight = sampling_rates[j])    
		predictions = model.predict(data_val)
		oob_error = 1 - model.oob_score_
		oob_error_data_sampling.append(oob_error)
		print("Forest Size: "+ str(forest_size[i])+ " Data Sampling Rate: " + str(sampling_rates[j]) + " Out of Bag Error: " + str(oob_error))
	plt.plot(sampling_rates, oob_error_data_sampling, label="Forest = "+str(forest_size[i]))

plt.xlabel("Data Sampling Rates")
plt.ylabel("Out of Bag Error Rate")
plt.legend()
plt.show()