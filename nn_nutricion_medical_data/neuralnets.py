#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 02:51:15 2020

@author: Isidro Gómez-Vargas
"""

import pandas as pd
# NaN values
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense
# import keras
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
from tensorflow.keras.models import Sequential
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Cargando los datos
try:
    data = pd.read_csv('data/breastRed.csv')
except:
    data = pd.read_csv('../data/breastRed.csv')

print(data.head())
# restan 26 columnas
print(data.info())
# X tiene todos los datos excepto el ultimo
x = data.loc[:, data.columns.isin(['Edad', 'IMC_categorico',
                                   'EdoHormonal', 'EtapaClinicaInicial',
                                   'Clasificacion', 'Histologia',
                                   'SBR', 'SobreexpresiónHer2neu',
                                   'TamanoTumor', 'PermeacionLinfovascular',
                                   'InvacionPerineural', 'ClasificacionMolecularCancer'])]

# catetegorias [df that will be used as "y"]
TipoCirugia = data.loc[:, 'TipoCirugia']
#
#TipoPrimerTx = data.loc[:, 'TipoPrimerTx']
#
#ReceptoresEstrogeno = data.loc[:, 'ReceptoresEstrogeno']
#
#ReceptoresProgesterona = data.loc[:, 'ReceptoresProgesterona']
#
#RecibioQT = data.loc[:, 'RecibioQT']
#
#ModalidadQT = data.loc[:, 'ModalidadQT']
#
#RecibioRT = data.loc[:, 'RecibioRT']
#
#Recibiohormonoterapia = data.loc[:, 'Recibiohormonoterapia']
#
#Recurrencia = data.loc[:, 'Recurrencia']
#
#TipoRecurrencia = data.loc[:, 'TipoRecurrencia']
#
#EstadoPaciente = data.loc[:, 'EstadoPaciente']

# Analogo: y = dataset.iloc[:, -1].values
X = x.values
# NaN values

imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
# imputer = SimpleImputer(strategy='median')

# For Imputer class axis 0 columns, 1 rows

imputer.fit(X[:, 1:10])
X[:, 1:10] = imputer.transform(X[:, 1:10])
x2 = pd.DataFrame(X)

# Encoding categorical data
# Encoding the Independent Variable
# labelencoder_X = LabelEncoder()
le = LabelEncoder()
# X['IMC_categorico'] = le.fit_transform(X.IMC_categorico.values)
X[:, 0] = le.fit_transform(X[:, 0])
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# categlist = [3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
categlist = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
onehotencoder = OneHotEncoder(categorical_features=categlist)
X = onehotencoder.fit_transform(X).toarray()
x3 = pd.DataFrame(X)

# y = TipoCirugia
X_train, X_test, y_train, y_test = train_test_split(X, TipoCirugia, test_size=0.3)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
x4 = pd.DataFrame(X_train)

# Red neuronal
classifier = Sequential()
# Hidden layers
# classifier.add(Dense(44, input_dim=87, activation='relu'))
# classifier.add(Dense(44, activation='relu'))
# tip: # nodes hidden layer can be de average(input, output)
classifier.add(Dense(31, input_dim=61, activation='relu'))
classifier.add(Dense(31, activation='relu'))

# Final layer
# classifier.add(Dense(4, activation = 'softmax'))
classifier.add(Dense(2, activation='softmax'))

# compiling
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
# classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# classifier.compile(loss='mean_absolute_error')

onehotencodery = OneHotEncoder(handle_unknown='ignore')

ytest = pd.DataFrame(y_test)
ytest = onehotencodery.fit_transform(ytest).toarray()
ytestpd = pd.DataFrame(ytest)

ytrain = pd.DataFrame(y_train)


#y_binary = to_categorical(y_train)
#ybinary = pd.DataFrame(y_binary)
labelencoder = LabelEncoder()
y_train2 = labelencoder.fit_transform(ytrain[['TipoCirugia']])
y_train2pd = pd.DataFrame(y_train2)

# classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fitting ann to training test
classifier.fit(X_train, y_train2, batch_size=64, nb_epoch=100)
# classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 100)

# predictor
y_pred = classifier.predict(X_test)
ypred = pd.DataFrame(y_pred)
y_pred = (y_pred > 0.5)
ypredd = pd.DataFrame(y_pred)
# output_dim = 44,  init = 'uniform', activation = 'relu', units = 87 ))

# Confusion matrix
ytest = (ytest > 0.5)
ytestpdf = pd.DataFrame(ytest)

    #y_test.values.argmax(axis=1), predictions.argmax(axis=1))
cm = confusion_matrix(ytestpdf.values.argmax(axis=1), ypredd.values.argmax(axis=1))

