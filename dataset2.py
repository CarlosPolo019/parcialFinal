# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:56:24 2021

@author: camae
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



def metricas_entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


def matriz_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr


def show_metrics(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))

url = 'diabetes.csv'
data = pd.read_csv(url)

# Tratamiento de datos

#data.job.replace(['blue-collar','admin.','management','technician','admin','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'], [0,1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
#data.housing.replace(['no','yes'], [0, 1], inplace=True)
#data.marital.replace(['married','single','divorced'], [0, 1,2], inplace=True)
#data.drop(['education', 'default', 'balance', 'loan', 'contact','day', 'month', 'duration', 'campaign','pdays', 'previous', 'poutcome', 'y'], axis = 1, inplace=True)
data['Age'].mean()
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.dropna(axis=0, how='any', inplace=True)

#x = np.array(data.drop(['housing'], 1))
#y = np.array(data.housing)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


data_train = data[:550]
data_test = data[550:]

# X y Y de la primera parte de dataset
x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) #0 Murió # 1 Vivió

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) #0 NO # 1 YES


# Maquinas de Soporte Vectorial
svc = SVC(gamma= 'auto')
svc.fit(x_train, y_train)
print('Maquina de soporte Vectorial')
print(f'Accuracy de Entrenamiento: {svc.score(x_train, y_train)}')
print(f'Accuracy de Test: {svc.score(x_test, y_test)}')

# Soporte Vectorial con validación cruzada
svc = SVC(probability=True)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    svc.fit(x_train[train],y_train[train])
    scores = svc.score(x_train[test], y_train[test])
    cvscores.append(scores)

print('Soporte Vectorial validación Cruzada')
print(f'Accuracy de Entrenamiento: {svc.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {svc.score(x_test, y_test)}')
y_pred = svc.predict(x_test_out);
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
matriz_confusion_svc = confusion_matrix(y_test_out, y_pred)
sns.heatmap(matriz_confusion_svc)
print(confusion_matrix(y_test_out, y_pred))
probs = svc.predict_proba(x_test)
probs = probs[:, 1]
fpr1, tpr1, _ = roc_curve(y_test, probs)

AUC = roc_auc_score(y_test, probs)
print(f'AUC: {AUC}')
print('*'* 50)

# Regresión Logística
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
logreg.fit(x_train, y_train)

print('Regresión Logística')
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')

# Regresión Logística con validación cruzada
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train],y_train[train])
    scores = logreg.score(x_train[test], y_train[test])
    cvscores.append(scores)

print('Regresión Logística validación Cruzada')
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
y_pred = logreg.predict(x_test_out);
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
matriz_confusion_logreg = confusion_matrix(y_test_out, y_pred)
sns.heatmap(matriz_confusion_logreg)

print(confusion_matrix(y_test_out, y_pred))
probs = logreg.predict_proba(x_test)
probs = probs[:, 1]
fpr2, tpr2, _ = roc_curve(y_test, probs)

AUC = roc_auc_score(y_test, probs)
print(f'AUC: {AUC}')
print('*'* 50)

# Vecinos mas cercanos clasificador
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

print('Clasificador Vecinos mas cercanos')
print(f'Accuracy de Entrenamiento: {knn.score(x_train, y_train)}')
print(f'Accuracy de Test: {knn.score(x_test, y_test)}')

# Vecinos mas cercanos clasificador con validación cruzada
knn = KNeighborsClassifier(n_neighbors=3)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    knn.fit(x_train[train],y_train[train])
    scores = knn.score(x_train[test], y_train[test])
    cvscores.append(scores)

print('Clasificador Vecinos mas cercanos validación Cruzada')
print(f'Accuracy de Entrenamiento: {knn.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {knn.score(x_test, y_test)}')
y_pred = knn.predict(x_test_out);
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
matriz_confusion_knn = confusion_matrix(y_test_out, y_pred)
sns.heatmap(matriz_confusion_knn)

print(confusion_matrix(y_test_out, y_pred))
probs = knn.predict_proba(x_test)
probs = probs[:, 1]
fpr3, tpr3, _ = roc_curve(y_test, probs)
AUC = roc_auc_score(y_test, probs)
print(f'AUC: {AUC}')
print('*'* 50)

# DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

print('DecisionTreeClassifier')
print(f'Accuracy de Entrenamiento: {dtc.score(x_train, y_train)}')
print(f'Accuracy de Test: {dtc.score(x_test, y_test)}')

# DecisionTreeClassifier con validación cruzada
dtc = DecisionTreeClassifier()
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    dtc.fit(x_train[train],y_train[train])
    scores = dtc.score(x_train[test], y_train[test])
    cvscores.append(scores)

print('DecisionTreeClassifier validación Cruzada')
print(f'Accuracy de Entrenamiento: {dtc.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {dtc.score(x_test, y_test)}')
y_pred = dtc.predict(x_test_out);
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
matriz_confusion_dct = confusion_matrix(y_test_out, y_pred)
sns.heatmap(matriz_confusion_dct)
plt.show()
print(confusion_matrix(y_test_out, y_pred))
probs = dtc.predict_proba(x_test)
probs = probs[:, 1]
fpr4, tpr4, _ = roc_curve(y_test, probs)
AUC = roc_auc_score(y_test, probs)
print(f'AUC: {AUC}')
print('*'* 50)
arrFpr = np.array([fpr4,fpr1, fpr2, fpr3])
arrFpt = np.array([tpr4,tpr1,tpr2,tpr3])
color = np.array(["green","blue","orange","red"])
for x in range(4):
    plt.plot(arrFpr[x], arrFpt[x], color=color[x],label=f'ROC {x+1}')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#Tuve problemas
#show_metrics('Decision Tree',AUC, accuracy_score(y_pred, y_test_out),dtc.score(x_test, y_test) , y_test, y_pred)
