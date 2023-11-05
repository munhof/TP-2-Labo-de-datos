# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Facundo Munho)s
"""
import Limpieza_de_datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importamos el modulo pyplot, y lo llamamos plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#importo el data set
cod_prendas = pd.read_csv("./Dataset-original/cod-prendas.csv")

book_fotos = Limpieza_de_datos.book_fotos()
# grafico el promedio de las remeras
dataset_shirt = book_fotos[0]
prom_shirt = dataset_shirt.drop(columns = 'label').mean().to_numpy()
imagen_prom_shirt = prom_shirt.reshape([28,28])
plt.matshow(imagen_prom_shirt)

# grafico el promedio de los pantalones
dataset_trouser = book_fotos[1]
prom_trouser = dataset_trouser.drop(columns = 'label').mean().to_numpy()
imagen_prom_trouser = prom_trouser.reshape([28,28])
plt.matshow(imagen_prom_trouser)

# grafico la diferencia absoluta entre el prom. de las dos imagenes
# esto me muestra las zonas de que mas diferencian una remeras de un pantalón
prom_shirt_trouser = prom_shirt-prom_trouser
promedio_shirt_trouser = np.absolute(prom_shirt_trouser)
imagen_prom_st = promedio_shirt_trouser.reshape([28,28])
plt.matshow(imagen_prom_st)


# Creo un dataframe con los datos de remeras y pantalones
dataRemeras = book_fotos[0]
dataPantalones = book_fotos[1]
dataRemPant = pd.concat([dataRemeras,dataPantalones])

#Verifico que esten balanceados
dataRemeras.info()
dataPantalones.info()
dataRemPant.info()

# Selecciono los pixeles de interes
print(promedio_shirt_trouser.max())
print(np.where(promedio_shirt_trouser >= 140))

# pixeles de interes
#  pixeles_brazos : [203,231,232,246,260]
#  pixeles_ombligo : [575,603,631,659,687,743]

# me quedo con los pixeles de interes y los sumo para asignarlos a los atributos brazo y ombligo
def transfromadorBrazoOmbligo(data):
    data['pixeles_brazos'] = data[['pixel203','pixel231','pixel232','pixel246','pixel260']].sum(axis=1)
    data['pixeles_ombligo'] = data[['pixel575','pixel603','pixel631','pixel659','pixel687','pixel743']].sum(axis=1)
    data = pd.DataFrame(data[['label','pixeles_brazos','pixeles_ombligo']])
    return data

dataRemerasPantalones = transfromadorBrazoOmbligo(dataRemPant)


# Grafico las prendas
sns.scatterplot(data = dataRemerasPantalones, x= 'pixeles_brazos', y = 'pixeles_ombligo',hue = 'label')
