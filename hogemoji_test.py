from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# print(digits.data)
# print(digits.data.shape)
n = len(digits.data)

# images = digits.images
# labels = digits.target
# for i in range(10):
#   plt.subplot(2, 5, i + 1)
#   plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#   plt.axis("off")
#   plt.title("Training: " +  str(labels[i]))
# plt.show()

# SVM
clf = svm.SVC(gamma=0.001, C=100.0)
# SVM_input
clf.fit(digits.data[:n*6//10 ], digits.target[:n*6//10])

# print(digits.target[-10:])
# print(clf.predict(digits.data[-10:]))

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
plt.show()

