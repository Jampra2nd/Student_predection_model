import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# allowing accesses to data set file
data = pd.read_csv("student-mat.csv", sep=";")

# trimming the data
data = data[["G1", "G2", "G3", "studytime", "absences", "failures", "health", "age"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# training station
'''''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
    with open("studentmodel.pickle", "wb")as f:
        pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predections = linear.predict(x_test)

# printing results from testing data
for x in range(len(predections)):
    print(predections[x], x_test[x], y_test[x])

# making graph
p = "health"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()