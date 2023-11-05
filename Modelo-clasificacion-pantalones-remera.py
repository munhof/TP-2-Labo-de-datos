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
    
Modelo KNN para pantalones y remeras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import Limpieza_de_datos

#en este codigo vamos a probar varios metodos de ajuste con knn con el cual
#vamos a evaluar la implementacion de varios algoritmos

#%% Distancia a los arquetipos
#una de las ideas es pensar que las imagenes forman un espacio espacio vectorial
#y cada clase de ropa es un sub-espacio de este, tomando a al arquetipo de prendas
#como referencia, la idea es clasificar segun la distancia alos arquetipos, pero vemos si 
#esto tiene sentido

#tomando toda la data de pantalones  y remeras y las grafico en relacion a ambas distancias

data_pantalon_remera = pd.concat([Limpieza_de_datos.book_fotos()[0],Limpieza_de_datos.book_fotos()[1]])
remera_arquetipo = Limpieza_de_datos.prendas_arquetipicas()[0]
pantalon_arquetipo = Limpieza_de_datos.prendas_arquetipicas()[1]

def distancia_arquetipo(data_frame):
    distancias_remera = []
    distancias_pantalon = []
    n = np.shape(data_frame)[0]
    for i in range(n):
        data_prenda = data_frame.iloc[i]
        array_prenda = data_prenda[1:785].to_numpy()
        arra_arquetipo_remera = remera_arquetipo.to_numpy()
        arra_arquetipo_pantalon = pantalon_arquetipo.to_numpy()
        #hago que la intensidad este entre 0 y 1  dividiendo por 255 para
        #asi tener numero mas manejables
        distancias_remera.append(np.linalg.norm(array_prenda/255-arra_arquetipo_remera/255,-2))
        distancias_pantalon.append(np.linalg.norm(array_prenda/255-arra_arquetipo_pantalon/255,-2))
    return distancias_remera, distancias_pantalon

distancia_remera,distancia_pantalon = distancia_arquetipo(data_pantalon_remera)
data_pantalon_remera["distancia remera"] = distancia_remera
data_pantalon_remera["distancia pantalon"] = distancia_pantalon

plt.plot()
sns.scatterplot(data = data_pantalon_remera, x = "distancia remera", y = "distancia pantalon",
                hue = "label")
plt.title("Distancia a arquetipos")
plt.show()

plt.plot()
sns.scatterplot(data = data_pantalon_remera, y = "label", x = "distancia pantalon",
                hue = "label")
plt.title("label vs distancia pantalon")
plt.show()


plt.plot()
sns.scatterplot(data = data_pantalon_remera, y = "label", x = "distancia remera",
                hue = "label")
plt.title("label vs distancia remera")
plt.show()

