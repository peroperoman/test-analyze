# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 各種演算
digits = datasets.load_digits()
n = len(digits.data)
clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(digits.data[:n*6//10 ], digits.target[:n*6//10])
expected = digits.target[-n*4//10:]
predicted = clf.predict(digits.data[-n*4//10:])
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
images = digits.images[-n*4//10:]

for i in range(12):
  plt.subplot(3, 4, i + 1)
  plt.axis("off")
  plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
  plt.title("Guess: " + str(predicted[i]))
plt.show()# coding: UTF-8
