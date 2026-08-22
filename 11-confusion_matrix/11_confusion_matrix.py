from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.model_selection import train_test_split

y_real = np.array([0, 0, 0, 0, 0, 1, 1])
y_prd = np.array([0, 0, 0, 0, 1, 0, 1])

confusion_m = confusion_matrix(y_real, y_prd)
print(confusion_m)

report = classification_report(y_real, y_prd)

print(report)

