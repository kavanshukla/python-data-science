import numpy as np
import scipy 
import sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import metrics
import matplotlib.pyplot as plt

data = np.load('forest_data.npz')
label_training = np.array(data['label_training'])
data_training = scale(data['data_training'])
data_val = data['data_val']
label_val = data['label_val']
print label_training

accuracy_results = []
c_list = [pow(10,-3), pow(10,-2), pow(10,10)]

Linearsvc_parameters = [{'C': c_list}]
model = GridSearchCV(LinearSVC(), Linearsvc_parameters ,cv=3, verbose=4)
model.fit(data_training, label_training)
predictions = model.predict(data_val)
accuracy = metrics.accuracy_score(label_val, predictions)

print(model.cv_results_)

split0 = model.cv_results_['split0_test_score']
split1 = model.cv_results_['split1_test_score']
split2 = model.cv_results_['split2_test_score']

print split0
print split1
print split2

for i in range(0,len(c_list)):
	print("C: " + str(c_list[i]) + " Split: 0" + " Score: " + str(split0[i]))
	print("C: " + str(c_list[i]) + " Split: 1" + " Score: " + str(split1[i]))
	print("C: " + str(c_list[i]) + " Split: 2" + " Score: " + str(split2[i]))
	plt.plot([1,2,3],[split0[i],split1[i],split2[i]], label="C = "+str(c_list[i]))

print("Final Test Accuracy: " + str(accuracy))
plt.xlabel("Validation Iteration Number")
plt.ylabel("Accuracy")
plt.title("Linear SVM")
plt.legend()
plt.show()
