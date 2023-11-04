# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Facundo Munho)s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inline_sql import sql, sql_val
# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt
import seaborn as sns

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



#creo un diccionario que guarda por separado las imagenes de cada tipo
    
book_fotos = {}
for cod in codigos:
    book_fotos[cod] = dataset[dataset["label"] == cod].drop(columns = ["label"])


#analizo la promedio de intensidad de un pixel en todo un tipo de prendas
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda =book_fotos[cod]
    primera_fila = dataset_prenda.mean().to_numpy()
    primera_imagen = primera_fila.reshape([28,28])
    ax.imshow(primera_imagen)
    
fig.show()

#el maximo
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda =book_fotos[cod]
    primera_fila = dataset_prenda.max().to_numpy()
    primera_imagen = primera_fila.reshape([28,28])
    ax.imshow(primera_imagen)
    
fig.show()

#el minimo
fig, axs = plt.subplots(nrows=10, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda =book_fotos[cod]
    primera_fila = dataset_prenda.min().to_numpy()
    primera_imagen = primera_fila.reshape([28,28])
    ax.imshow(primera_imagen)
    
fig.show()

#del caso del maximo de aparicion, entre todas las prendas puedo eliminar 
#los casos que nunca se utiliza a un pixel (si su valor maximo entre todas
#las fotos es 0 entonces nunca se utiliza y no aporta informacion)


maximos_prendas = dataset.max().array.to_numpy()[1:].reshape([28,28])
plt.matshow(maximos_prendas)
plt.title("Maxima intensidad prendas")

minimo_prendas = dataset.min().array.to_numpy()[1:].reshape([28,28])
plt.matshow(minimo_prendas)
plt.title("minima intensidad prendas")

promedio_prendas = dataset.mean().array.to_numpy()[1:].reshape([28,28])
plt.matshow(promedio_prendas)
plt.title("promedia intensidad prendas")


media_prendas = dataset.median().array.to_numpy()[1:].reshape([28,28])
plt.matshow(media_prendas)
plt.title("media intensidad prendas")

desviacion_pix = dataset.std().array.to_numpy()[1:].reshape([28,28])
plt.matshow(desviacion_pix)
plt.title("std intensidad prendas")

#grafico de la media,moda,std y maximo de cada prenda

fig, axs = plt.subplots(nrows=10, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})



codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)
grfico_title = ["media","moda","std","maximo"]

for i in range(10):
    dataset_prenda = book_fotos[i]
    dataset_prenda.head()
    for j in range(4):
        axs[i,j].set_title(str(prendas[i] + grfico_title[j]))
        if j == 0:
            primera_fila = dataset_prenda.mean().to_numpy()
        elif j == 1:
            primera_fila = dataset_prenda.median().to_numpy()
        elif j == 2:
            primera_fila = dataset_prenda.std().to_numpy()
        else:
            primera_fila = dataset_prenda.max().to_numpy()
        primera_imagen = primera_fila.reshape([28,28])
        axs[i,j].imshow(primera_imagen)



# Analizo la variabilidad de los datos dentro de cada clase y las comparo entre si.

desviaciones = np.ones(10)  # Placeholder

for i in range (10):
    desviacion_total_por_clase = dataset[dataset['label'] == i ].std().array.to_numpy()[1:].sum()
    # Agarro una clase, calculo la desviacion de cada columna y luego sumo las columnas.
    desviaciones[i] = desviacion_total_por_clase

desviacion_minima = np.min(desviaciones)
desviacion_normalizada = desviaciones/desviacion_minima # Por esto use array, para poder normalizar facilmente.

nombre_de_prendas = np.array(cod_prendas['Tipo De Prenda'])
dataset = pd.DataFrame({'nombre_de_prendas': nombre_de_prendas, 'desviacion_normalizada': list(desviacion_normalizada)}, columns=['nombre_de_prendas', 'desviacion_normalizada'])
dataset = dataset.sort_values('desviacion_normalizada')
sns.scatterplot(data = dataset , y = 'nombre_de_prendas', x = 'desviacion_normalizada').set(title = 'Variabilidad normalizada de los datos de cada clase')
plt.show()
plt.close()
# Con esta metrica la variabilidad entre los vestidos es alrededor de un 33% mas grande que entre sneakers, 
# que son los que menos variabilidad tienen.
