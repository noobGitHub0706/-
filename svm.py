from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics

#データの準備
digits = datasets.load_digits()
n_samples = len(digits.data)
print("データ数：{}".format(n_samples))
#データの可視化
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
 plt.subplot(2, 5, index + 1)
 plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
 plt.axis('off')
 plt.title('Training: %i'% label)
plt.show()

# SVM の読み込み
clf = svm.SVC(gamma=0.001, C=100.)

# 60%のデータで学習実行
clf.fit(digits.data[:int(n_samples * 6 / 10)], digits.target[:int(n_samples * 6 / 10)])

# 40%のデータでテスト
expected = digits.target[int(n_samples *-4 / 10):]
predicted = clf.predict(digits.data[int(n_samples *-4 / 10):])
print("Classification report for classifier %s:¥n%s¥n" % (clf,metrics.classification_report(expected, predicted)))
print("Confusion matrix:¥n%s" % metrics.confusion_matrix(expected, predicted))

#予測結果を可視化
images_and_predictions = list(zip(digits.images[int(n_samples *-4 / 10):], predicted))
for index,(image, prediction) in enumerate(images_and_predictions[:12]):
 plt.subplot(3, 4, index + 1)
 plt.axis('off')
 plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
 plt.title('Prediction: %i' % prediction)
plt.show()