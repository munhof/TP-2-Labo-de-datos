# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Facundo Munho)s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inline_sql import sql, sql_val
# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt

#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#importo el data set
dataset = pd.read_csv("./Dataset-original/fashion-mnist.csv")
cod_prendas = pd.read_csv("./Dataset-original/cod-prendas.csv")

print("Las columnas del data frame son")
print(dataset.columns)
print("Los valores unicos de label son")
print(dataset.label.unique())
print("Los valores unicos de alguna columnas pixel son")
print(dataset.pixel88.unique())


#si veo el tamaño del data set, tengo que es de 60000x785
#es decir tenemos las 60000 prendas con sus imagenes de 24x24pixeles (784)
#y el label identificatorio, que se encuentra en la primer columnas del atributo

#el codigo de las prendas es 
#0 T-shirt/top
#1 Trouser
#2 Pullover
#3 Dress
#4 Coat
#5 Sandal
#6 Shirt
#7 Sneaker
#8 Bag
#9 Ankle boot

#que estan cargados en el csv  cod-prendas

#abro una imagen para ver como se comporta, para eso tomo la primera fila

primera_fila = dataset.loc[0].to_numpy()
print("El tipo de la prenda es")
print(primera_fila[0])
print("Que es Pullover")

#el primer elemento del array es el tipo, que es pullover
primer_tipo = primera_fila[0]

primera_imagen = primera_fila[1:].reshape([28,28])
plt.matshow(primera_imagen)

#grafico uno de cada uno
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda = dataset[dataset["label"] == cod]
    primera_fila = dataset_prenda.iloc[0].to_numpy()
    primera_imagen = primera_fila[1:].reshape([28,28])
    ax.imshow(primera_imagen)
    
fig.show()

#otra informacion interesante es saber cuantas de cada prenda hay cargadas

for cod, prenda in zip(codigos, prendas):
    dataset_prenda = dataset[dataset["label"] == cod]
    print("De " + prenda + " hay cargadas:" ,  dataset_prenda.shape[0])

#hay la misma cantidad cantidad de cada uno

