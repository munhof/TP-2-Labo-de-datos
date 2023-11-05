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

# configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'
# importo el data set
dataset = pd.read_csv("./Dataset-original/fashion-mnist.csv")
cod_prendas = pd.read_csv("./Dataset-original/cod-prendas.csv")
# %%

# analisamos el data set como conjunto
print("Las columnas del data frame son")
print(dataset.columns)
print("Los valores unicos de label son")
print(dataset.label.unique())
print("Los valores unicos de alguna columnas pixel son")
print(dataset.pixel88.unique())


# si veo el tamaño del data set, tengo que es de 60000x785
# es decir tenemos las 60000 prendas con sus imagenes de 24x24pixeles (784)
# y el label identificatorio, que se encuentra en la primer columnas del atributo

# el codigo de las prendas es
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

# que estan cargados en el csv  cod-prendas

# abro una imagen para ver como se comporta, para eso tomo la primera fila

primera_fila = dataset.loc[0].to_numpy()
print("El tipo de la prenda es")
print(primera_fila[0])
print("Que es Pullover")

# el primer elemento del array es el tipo, que es pullover
primer_tipo = primera_fila[0]

primera_imagen = primera_fila[1:].reshape([28, 28])
plt.matshow(primera_imagen)

plt.savefig("./figuras-exploracion-de-datos/primera-prenda-dataset.png")
plt.close()
del primer_tipo, primera_imagen

# grafico uno de cada uno
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda = dataset[dataset["label"] == cod]
    primera_fila = dataset_prenda.iloc[0].to_numpy()
    primera_imagen = primera_fila[1:].reshape([28, 28])
    ax.imshow(primera_imagen)

plt.savefig("./figuras-exploracion-de-datos/primeras-prendas-de-cada-clase.png")
plt.close()

del ax, prenda, cod, axs, fig, primera_fila, primera_imagen

# otra informacion interesante es saber cuantas de cada prenda hay cargadas

for cod, prenda in zip(codigos, prendas):
    dataset_prenda = dataset[dataset["label"] == cod]
    print("De " + prenda + " hay cargadas:",  dataset_prenda.shape[0])

# hay la misma cantidad cantidad de cada uno
del cod, prenda, codigos, prendas
# %%
# Aca analisamos como de comporta la informacion de cada pixel
# tanto en conjunto como en cada grupo de prendas

# creo un diccionario que guarda por separado las imagenes de cada tipo
codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)

book_fotos = {}
for cod in codigos:
    book_fotos[cod] = dataset[dataset["label"] == cod].drop(columns=["label"])


# analizo la promedio de intensidad de un pixel en todo un tipo de prendas
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda = book_fotos[cod]
    promedio_intensidad_pixel = dataset_prenda.mean().to_numpy()
    imagen_promedio_intensidad = promedio_intensidad_pixel.reshape([28, 28])
    ax.imshow(imagen_promedio_intensidad)

fig.suptitle("media de intensidad de" + prenda)
fig.show()
plt.savefig("./figuras-exploracion-de-datos/imagen-de-promedio-de-intensidad.png")
plt.close()

del ax, prenda, cod, axs, fig, promedio_intensidad_pixel, imagen_promedio_intensidad

# el maximo
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda = book_fotos[cod]
    maxima_intensidad_pixel = dataset_prenda.max().to_numpy()
    imagen_maxima_intensidad = maxima_intensidad_pixel.reshape([28, 28])
    ax.imshow(imagen_maxima_intensidad)

fig.suptitle("maxima de intensidad de" + prenda)
fig.show()
plt.savefig("./figuras-exploracion-de-datos/imagen-de-maxima-de-intensidad.png")
plt.close()
del ax, prenda, cod, axs, fig, maxima_intensidad_pixel, imagen_maxima_intensidad

# el minimo
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

codigos = cod_prendas["Codigo"].array.to_numpy().reshape(10)
prendas = cod_prendas["Tipo De Prenda"].array.to_numpy().reshape(10)


for ax, cod, prenda in zip(axs.flat, codigos, prendas):
    ax.set_title(str(prenda))
    dataset_prenda = book_fotos[cod]
    minima_intensidad_pixel = dataset_prenda.min().to_numpy()
    imagen_minima_intensidad = minima_intensidad_pixel.reshape([28, 28])
    ax.imshow(imagen_minima_intensidad)

fig.suptitle("minima intensidad de" + prenda)
fig.show()
plt.savefig("./figuras-exploracion-de-datos/imagen-de-minima-de-intensidad.png")
plt.close()
del ax, prenda, cod, axs, fig, minima_intensidad_pixel, imagen_minima_intensidad

# del caso del maximo de aparicion, entre todas las prendas puedo eliminar
# los casos que nunca se utiliza a un pixel (si su valor maximo entre todas
# las fotos es 0 entonces nunca se utiliza y no aporta informacion)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 30),
                        subplot_kw={'xticks': [], 'yticks': []})

maximos_prendas = dataset.max().array.to_numpy()[1:].reshape([28, 28])
axs[0].matshow(maximos_prendas)
axs[0].set_title("Maxima intensidad prendas")

minimo_prendas = dataset.min().array.to_numpy()[1:].reshape([28, 28])
axs[1].matshow(minimo_prendas)
axs[1].set_title("Minima intensidad prendas")

fig.suptitle("Maximo y minima intensidad en las prendas")
plt.show()
plt.savefig("./figuras-exploracion-de-datos/maximo-y-minimo-de-intensidad.png")
plt.close()
del axs, fig, maximos_prendas, minimo_prendas

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 30),
                        subplot_kw={'xticks': [], 'yticks': []})

promedio_prendas = dataset.mean().array.to_numpy()[1:].reshape([28, 28])
axs[0].matshow(promedio_prendas)
axs[0].set_title("Media de intensidad prendas")

media_prendas = dataset.median().array.to_numpy()[1:].reshape([28, 28])
axs[1].matshow(media_prendas)
axs[1].set_title("Mediana de intensidad prendas")

desviacion_pix = dataset.std().array.to_numpy()[1:].reshape([28, 28])
axs[2].matshow(desviacion_pix)
axs[2].set_title("Desviacion estandar de intensidad prendas")

fig.suptitle("Medidas resumen de intensidad en las prendas")
plt.show()
plt.savefig("./figuras-exploracion-de-datos/medidas-resumen-intensidad.png")
plt.close()
del axs, fig, promedio_prendas,media_prendas,desviacion_pix

# grafico de la media,moda,std y maximo de cada prenda


grafico_title = ["media", "moda", "std", "maximo"]
for i in range(10):

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 30),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle("Medidas resumen de "+prendas[i])
    dataset_prenda = book_fotos[i]
    dataset_prenda.head()
    for j in range(4):
        axs[j].set_title(grafico_title[j])
        if j == 0:
            data_imagen = dataset_prenda.mean().to_numpy()
        elif j == 1:
            data_imagen = dataset_prenda.median().to_numpy()
        elif j == 2:
            data_imagen = dataset_prenda.std().to_numpy()
        else:
            data_imagen = dataset_prenda.max().to_numpy()
            
        imagen = data_imagen.reshape([28, 28])
        axs[j].imshow(imagen)
    plt.show()
    plt.savefig("./figuras-exploracion-de-datos/medidas-resumen-intensidad-por-prenda-"+ prendas[i] + ".png")
plt.close()
del i, j, axs, fig, data_imagen, imagen,grafico_title, dataset_prenda, codigos, prendas

# %%

# Analizo la variabilidad de los datos dentro de cada clase y las comparo entre si.

desviaciones = np.ones(10)  # Placeholder

for i in range(10):
    desviacion_total_por_clase = book_fotos[i].std().array.to_numpy()[1:].sum()
    # Agarro una clase, calculo la desviacion de cada columna y luego sumo las columnas.
    desviaciones[i] = desviacion_total_por_clase

#normalizo las desviaciones
desviacion_maxima = np.max(desviaciones)
desviacion_normalizada = desviaciones/desviacion_maxima

nombre_de_prendas = cod_prendas['Tipo De Prenda'].to_numpy()

desviaciones_por_prendas = pd.DataFrame({'nombre_de_prendas': nombre_de_prendas, 'desviacion_normalizada': list(
    desviacion_normalizada)}, columns=['nombre_de_prendas', 'desviacion_normalizada'])

desviaciones_por_prendas = desviaciones_por_prendas.sort_values('desviacion_normalizada')

fig = plt.figure(figsize=(10,5))
plt.grid()
sns.scatterplot(data=desviaciones_por_prendas, x='nombre_de_prendas', y='desviacion_normalizada').set(
    title='Variabilidad normalizada de los datos de cada clase')
plt.show()
plt.savefig("./figuras-exploracion-de-datos/variabilidad-de-prendas.png")
plt.close()
# Con esta metrica la variabilidad entre los vestidos es alrededor de un 33% mas grande que entre sneakers,
# que son los que menos variabilidad tienen.
