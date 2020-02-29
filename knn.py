import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as numpy
from sklearn import linear_model, preprocessing

data = pd.read_csv("car_data.txt", sep=",")

e = preprocessing.LabelEncoder()

buying = e.fit_transform(list(data["buying"]))
maint = e.fit_transform(list(data["maint"]))
door = e.fit_transform(list(data["door"]))
persons = e.fit_transform(list(data["persons"]))
lug_boot = e.fit_transform(list(data["lug_boot"]))
safety = e.fit_transform(list(data["safety"]))
clas = e.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)



model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)


print(predicted)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
	print("predicted: ", names[predicted[x]], "data:", x_test[x], "Actual: ", names[y_test[x]])
