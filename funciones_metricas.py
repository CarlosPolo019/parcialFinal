#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:02:20 2020

@author: robertomorales
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

simplefilter(action='ignore', category=FutureWarning)


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


def show_roc_curve_matrix(fpr, tpr, matriz_confusion):
    sns.heatmap(matriz_confusion)
    plt.show()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def show_metrics(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')


url = 'titanic.csv'
data = pd.read_csv(url)


data.Embarked.value_counts(dropna=False)
# data.info()

# Volver categoricos los datos
data.Sex.replace(['female', 'male'], [0, 1], inplace=True)
data.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)


# Reemplazar los nan
data.Age.replace(np.nan, 30, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)
data.Embarked = data.Embarked.astype(np.int)

x = np.array(data.drop(['Survived'], 1))
y = np.array(data.Survived)  # 0 Murió, 1 Vivió

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model, acc_validation, acc_test, y_pred = metricas_entrenamiento(model, x_train, x_test, y_train, y_test)
matriz_confusion, AUC, fpr, tpr  = matriz_confusion_auc(model, x_test, y_test, y_pred)
show_metrics('Linear Regression', AUC, acc_validation, acc_test, y_test, y_pred)
show_roc_curve_matrix(fpr, tpr, matriz_confusion)


model = DecisionTreeClassifier()
model, acc_validation, acc_test, y_pred =metricas_entrenamiento(model, x_train, x_test, y_train, y_test)
matriz_confusion, AUC, fpr, tpr  = matriz_confusion_auc(model, x_test, y_test, y_pred)
show_metrics('Decision Tree',AUC, acc_validation, acc_test, y_test, y_pred)
show_roc_curve_matrix(fpr, tpr, matriz_confusion)

model = KNeighborsClassifier(n_neighbors = 3)
model, acc_validation, acc_test, y_pred =metricas_entrenamiento(model, x_train, x_test, y_train, y_test)
matriz_confusion, AUC, fpr, tpr  = matriz_confusion_auc(model, x_test, y_test, y_pred)
show_metrics('KNeighborns',AUC, acc_validation, acc_test, y_test, y_pred)
show_roc_curve_matrix(fpr, tpr, matriz_confusion)






    
















