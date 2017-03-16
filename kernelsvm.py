import numpy as np
import scipy 
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import metrics

data = np.load('forest_data.npz')
label_training = np.array(data['label_training'])
data_training = scale(data['data_training'])
data_val = data['data_val']
label_val = data['label_val']
print label_training

accuracy_results = []
c_list = [pow(10,-3), pow(10,-2), pow(10,10)]
sigma_list = []
gamma_list = [pow(10,0), pow(10,2), pow(10,4)]

for i in gamma_list:
	sigma_list.append(1.0/2*np.square(i))

Kernelsvm_parameters = [{'C': c_list, 'gamma': sigma_list, 'kernel': ['rbf']}]
model = GridSearchCV(SVC(), Kernelsvm_parameters ,cv=3, verbose=4)
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

best_accuracy_list = []

for i in range(0,len(split0)):
	best_accuracy = max(split0[i], split1[i], split2[i])
	best_accuracy_list.append(best_accuracy)

m=0
for i in range(0,len(c_list)):
    accr = []
    for j in range(0,len(sigma_list)):
        print("C: " + str(c_list[i]) + " Sigma: " + str(sigma_list[j]) + " Accuracy: " + str(best_accuracy_list[m]))
        accr.append(best_accuracy_list[m])
        m = m + 1

print("Final Test Accuracy: " + str(accuracy))