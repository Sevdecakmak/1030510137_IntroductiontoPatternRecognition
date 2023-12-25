import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data set
data = pd.read_excel('otu1.xlsx')

# Convert left/right tags to 0/1
le = LabelEncoder()
data['Class'] = le.fit_transform(data['Sample1'])  # 0: Sol, 1: Sağ

# Separate attribute and target
X = data.iloc[:, 1:]
y = data['Class']

# Divide the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Use of cross validation
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
print('Çapraz doğrulama skorları:', scores)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Doğruluk:', accuracy)

# Confusion matrix
confusion = confusion_matrix(y_test, predictions)
print('Karışıklık matrisi: ')
print(confusion)

# Sensitivity, specificity and F1 score report
report = classification_report(y_test, predictions)
print('Sınıflandırma Raporu:')
print(report)

# ROC curve plotting
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Sensitivity and specificity
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
print('   Hassasiyet:', sensitivity)
print('   Özgüllük:', specificity)