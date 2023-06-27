# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:54:41 2022

@author: Usuario
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import *
from string import ascii_uppercase
import seaborn as sns
import csv

def extraer(nombre_archivo, indice_columna):
    numero_fila = 0
    with open(nombre_archivo, "r") as entrada:
        csv_reader = csv.reader(entrada, delimiter=',')
        for fila in csv_reader:
            yield numero_fila, fila[indice_columna]
            numero_fila += 1

clase = [x[1] for x in extraer("clases.csv", 5) if x[0] >= 2]
claseTabla = [x[1] for x in extraer("clases.csv", 6) if x[0] >= 2]
y_verd = np.array(clase)
y_pred = np.array(claseTabla)

y_verd = y_verd.astype('int')
y_pred = y_pred.astype('int')


confm = confusion_matrix(y_verd, y_pred)
print(confm)

columnas = ['Clase %s'%(i) for i in list(ascii_uppercase)[0:len(np.unique(y_pred))]]
df_cm = pd.DataFrame(confm, index=columnas,columns=columnas)
grafica = sns.heatmap(df_cm,cmap='Pastel1',annot=True)

Accuracy = accuracy_score(y_verd, y_pred)
Recall = recall_score(y_verd, y_pred, average=None)
Precision = precision_score(y_verd, y_pred, average=None)
F1 = f1_score(y_verd, y_pred, average=None)
print("Accuracy: ", Accuracy, "Recall: ", Recall, "Precision: ",Precision, "F1: ", F1)

c = classification_report(y_verd, y_pred)
print(c)
