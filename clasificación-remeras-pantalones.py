# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Facundo Munho)s
"""
from Exploracion_de_archivos import book_fotos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inline_sql import sql, sql_val
# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#importo el data set
dataset = pd.read_csv("fashion-mnist.csv")
cod_prendas = pd.read_csv("cod-prendas.csv")

# grafico el promedio de las camisetas
dataset_shirt = book_fotos[0]
prom_shirt = dataset_shirt.mean().to_numpy()
imagen_prom_shirt = prom_shirt.reshape([28,28])
plt.matshow(imagen_prom_shirt)

# grafico el promedio de los pantalones
dataset_trouser = book_fotos[1]
prom_trouser = dataset_trouser.mean().to_numpy()
imagen_prom_trouser = prom_trouser.reshape([28,28])
plt.matshow(imagen_prom_trouser)

# grafico la diferencia absoluta entre el prom. de las dos imágenes
# esto me muestra las zonas de que más diferencian una camiseta de un pantalón
prom_shirt_trouser = prom_shirt-prom_trouser
promedio_shirt_trouser = np.absolute(prom_shirt_trouser)
imagen_prom_st = promedio_shirt_trouser.reshape([28,28])
plt.matshow(imagen_prom_st)

# Selecciono los pixeles de interes
print(promedio_shirt_trouser.max())
print(np.where(promedio_shirt_trouser >= 140))

pixeles_brazos = [203,231,232,246,260]
pixeles_ombligo = [575,603,631,659,687,743]

dataPoleras = dataset[dataset['label']==0]
dataPantalones = dataset[dataset['label']==1]
dataPolPant = pd.concat([dataPoleras,dataPantalones])

dataPolPant['pixeles_brazos'] = dataPolPant[['pixel203','pixel231','pixel232','pixel246','pixel260']].sum(axis=1)
dataPolPant['pixeles_ombligo'] = dataPolPant[['pixel575','pixel603','pixel631','pixel659','pixel687','pixel743']].sum(axis=1)
dataPolerasPantalones = dataPolPant[['label','pixeles_brazos','pixeles_ombligo']]

# Grafico las prendas
sns.scatterplot(data = dataPolerasPantalones, x= 'pixeles_brazos', y = 'pixeles_ombligo', hue='label')