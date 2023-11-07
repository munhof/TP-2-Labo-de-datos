# -*- coding: utf-8 -*-
"""
Trabajo Practico 2
Alumnos:
    Munho Vital Facundo Nicolas
    Cáceres Blanco Juan Manuel
    Pavez Cayupe Richard Arturo
Grupo:
    Club penguin
Materia:
    Labo De Datos
Cuatrimestre y año:
    2c 2023
Departamento a cargo:
    DC
    
Modelo arboles de decision para clasificacion multiclase
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score

import Limpieza_de_datos

#en este codigo vamos a probar varios metodos de ajuste con knn con el cual
#vamos a evaluar la implementacion de varios algoritmos

#%% Distancia a los arquetipos
#una de las ideas es pensar que las imagenes forman un espacio espacio vectorial
#y cada clase de ropa es un sub-espacio de este, tomando a al arquetipo de prendas
#como referencia, la idea es clasificar segun la distancia alos arquetipos, pero vemos si 
#esto tiene sentido

#tomando toda la data de pantalones  y remeras y las grafico en relacion a ambas distancias

data_prendas = Limpieza_de_datos.dataset
arquetipos = Limpieza_de_datos.prendas_arquetipicas()

