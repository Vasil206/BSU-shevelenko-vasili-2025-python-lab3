
import pandas as pd

train = pd.read_csv("lab2/preprocessed.csv")
test = pd.read_csv("lab2/preprocessed_test.csv")

x_train = train.drop(['Transported_int'], axis='columns')
y_train = train['Transported_int']

x_test = test.drop(['Transported_int'], axis='columns')
y_test = test['Transported_int']


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)


y_pred_train = tree.predict(x_train)
y_pred_test = tree.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("test report")
report = classification_report(y_test, y_pred_test)
print(report)
print(confusion_matrix(y_test, y_pred_test))
print(f'Accuracy: {accuracy_score(y_test, y_pred_test):.2f}',"\n\n\n")

print("train report")
reportTrain = classification_report(y_train, y_pred_train)
print(reportTrain)
print(confusion_matrix(y_train, y_pred_train))
print(f'Accuracy: {accuracy_score(y_train, y_pred_train):.2f}',"\n\n\n")


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 8))
plot_tree(tree, filled=True)
plt.show()
