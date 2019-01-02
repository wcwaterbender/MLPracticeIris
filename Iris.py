import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

fileName = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(fileName, names=names)

#some data visualizations
#box-whisker
#dataset.plot(kind ='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#histograms
#dataset.hist()
#plt.show()

#scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

#create a validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size =0.20
seed = 8
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state = seed)


