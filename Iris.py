import pandas
import sklearn
import matplotlib.pyplot as plt

fileName = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(fileName, names=names)
print(dataset.head(20))
