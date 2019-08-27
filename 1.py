import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../tea/data/mushrooms.csv")

print(data.head())
# 8124 instance, 23 atributa

print(data.info())
# Nema nedostajucih vrednosti

# Jedinstvene vrednosti u skupu:
for col in data.columns.values:
    print ("{0}: {1}".format(col, data[col].unique()))

# Veil-type samo 'p', uklonicemo taj atribut, jer nema nikakav uticaj na klasifikaciju.
data = data.drop("veil-type", axis = 1)

data['class'].replace('p', 0, inplace = True)
data['class'].replace('e', 1, inplace = True)

data = pd.get_dummies(data)

print(data.head())

labels = data["class"].values
data = data.drop(["class"], axis = 1)

from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.3)

from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(data_train.shape[1], )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(data_train,labels_train,epochs=5,batch_size=80)

labels_pred = model.predict(data_test)
labels_pred = (labels_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

'''
test = model.evaluate(data_test, labels_test)
print(test)

model.save("first.h5")

'''
