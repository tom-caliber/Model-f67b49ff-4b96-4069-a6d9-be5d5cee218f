from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import dill as pickle

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

filename = 'model.pk'
with open(filename, 'wb') as file:
  pickle.dump(clf, file)
#joblib.dump(clf, 'model.pk')