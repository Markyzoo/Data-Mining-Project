
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt 

A_score = [0.73,0.69,0.67,0.55,0.47,0.45,0.44,0.35,0.15,0.08]
B_score = [0.61,0.03,0.68,0.31,0.45,0.09,0.38,0.05,0.01,0.04]

true_result = [1,1,0,0,1,1,0,0,1,0]

A_1_result = []
A_5_result = []

for x in A_score:
	if(x>0.5):
		A_5_result .append(1)
	else:
		A_5_result .append(0)

for x in A_score:
	if(x>0.1):
		A_1_result .append(1)
	else:
		A_1_result .append(0)

# print("B predictions are: ", B_result)	
# print("Confusion Matrix: ")
# print(sklearn.metrics.confusion_matrix(true_result,B_result))
# print("F measure: ", sklearn.metrics.f1_score(true_result,B_result))

# fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_result, A_1_result)
# plt.plot(fpr,tpr,label="0.1 threshold")

fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_result,A_score)
plt.plot(fpr, tpr,label="A")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.legend(loc=0)
locs, labels = plt.yticks()
plt.yticks(np.arange(0,1,step=0.1))
locs, labels = plt.xticks()
plt.xticks(np.arange(0,1,step=0.1))
plt.show()

