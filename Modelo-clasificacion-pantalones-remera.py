# -*- coding: utf-8 -*-
"""
Trabajo Practico 2
Alumnos:
    Munho Vital Facundo Nicolas
    Cáceres Blanco Juan Manuel
    Pavez Cayupel Richard Arturo
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve

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
        distancias_remera.append(np.linalg.norm(array_prenda/255-arra_arquetipo_remera/255))
        distancias_pantalon.append(np.linalg.norm(array_prenda/255-arra_arquetipo_pantalon/255))
    return distancias_remera, distancias_pantalon

distancia_remera,distancia_pantalon = distancia_arquetipo(data_pantalon_remera)
data_pantalon_remera["distancia remera"] = distancia_remera
data_pantalon_remera["distancia pantalon"] = distancia_pantalon

plt.plot()
sns.scatterplot(data = data_pantalon_remera, x = "distancia remera", y = "distancia pantalon",
                hue = "label")
plt.title("Distancia a arquetipos")
plt.show()

del distancia_pantalon, distancia_remera, data_pantalon_remera

#vemos que hay dos grandes regiones donde tiene sentido realizar la clasificacion
#con esta metrica, veamos que pasa con knn

#importo la data para el modelo

X_train, X_test, y_train, y_test = Limpieza_de_datos.data_train_modelo_pantalon_remera()

#les calculo las distancias a test y train (modifico la funcion pra datos sin label)
def distancia_arquetipo(data_frame):
    distancias_remera = []
    distancias_pantalon = []
    n = np.shape(data_frame)[0]
    for i in range(n):
        data_prenda = data_frame.iloc[i]
        array_prenda = data_prenda[:785].to_numpy()
        arra_arquetipo_remera = remera_arquetipo.to_numpy()
        arra_arquetipo_pantalon = pantalon_arquetipo.to_numpy()
        #hago que la intensidad este entre 0 y 1  dividiendo por 255 para
        #asi tener numero mas manejables
        distancias_remera.append(np.linalg.norm(array_prenda/255-arra_arquetipo_remera/255))
        distancias_pantalon.append(np.linalg.norm(array_prenda/255-arra_arquetipo_pantalon/255))
    return distancias_remera, distancias_pantalon

distancia_remera,distancia_pantalon = distancia_arquetipo(X_train)
X_train["distancia remera"] = distancia_remera
X_train["distancia pantalon"] = distancia_pantalon

distancia_remera,distancia_pantalon = distancia_arquetipo(X_test)
X_test["distancia remera"] = distancia_remera
X_test["distancia pantalon"] = distancia_pantalon

#ahora tomo los valores especificos con los cuales voy a trabajar
x_train = X_train[["distancia remera","distancia pantalon"]]
x_test = X_test[["distancia remera","distancia pantalon"]]

#armo el modelo, arbitrariamente elijo 5
knn_distancia = KNeighborsClassifier(n_neighbors=5) 

#entreno el modelo
knn_distancia.fit(x_train,y_train)


data_test = x_test.copy()
data_test["label"] = y_test.copy()

data_test_modelo = x_test.copy()
data_test_modelo["label"] = knn_distancia.predict(x_test)

#reviso el score con los dato de test
print(knn_distancia.score(x_test, y_test))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axs[0] = sns.scatterplot(data = data_test, x = "distancia remera", y = "distancia pantalon", 
                         hue = "label", ax = axs[0] )
axs[0].set_title("distribucion datos original")

axs[1] = sns.scatterplot(data = data_test_modelo, x = "distancia remera", y = "distancia pantalon",
                         hue = "label",  ax = axs[1] )
axs[1].set_title("distribucion datos modelo")
fig.suptitle("Modelo de clasficiacion de KNN por \n distancia arquetipos")

#borro las varibales
del fig,axs,data_test_modelo,data_test,knn_distancia,x_test,x_train,
del pantalon_arquetipo, remera_arquetipo
del distancia_pantalon, distancia_remera
#ahora comprobemos como se comporta el modelo variando el hyper parametro n_neighbors

x_train = X_train[["distancia remera","distancia pantalon"]]
x_test = X_test[["distancia remera","distancia pantalon"]]


#armo una lista de parametros, en este caso distruidos con la secuencia de fibonacci
n = 145  # Reemplaza 10 con el valor deseado de 'n'
fibonacci = [0,1]
fibonacci = [fibonacci[i - 1] + fibonacci[i - 2] for i in range(2, n) if (fibonacci := fibonacci + [fibonacci[-1] + fibonacci[-2]])[-1] < n]

#defino hyperparametros
hyper_params = {'n_neighbors': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]}

#defino el modelo
knn_distancia = KNeighborsClassifier()
clf = GridSearchCV(knn_distancia, hyper_params,cv = 5)
search = clf.fit(x_train, y_train)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param = search.best_params_["n_neighbors"]
mejor_score = search.best_score_
print(f"mejor score {search.best_score_}")

clf = KNeighborsClassifier(n_neighbors=5) 
clf.fit(x_train,y_train,)

fpr, tpr, _ = precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
plt.plot(fpr,tpr)
#sin embargo verifico la performance de cada uno para ver como se comporta

def evaluate_model(model, X_train,X_test, y_train,y_test, hyperparameters, cv):
    # Listas para almacenar los resultados
    score_best = []
    score_worse = []
    score_mean = []
    roc_curves = []
    roc_aucs = []


    for param in hyperparameters:
        clf = model(**param)
        # Calcula los puntajes de validación cruzada
        score_cv = cross_val_score(clf, X_train, y_train, cv=cv)
        score_best.append(max(score_cv))
        score_worse.append(min(score_cv))
        score_mean.append(score_cv.mean())
        
        # Listas para almacenar las predicciones y etiquetas reales
        clf.fit(X_train, y_train)
        
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        
            
        # Después de completar todos los folds
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # Calcula la curva ROC
        roc_curves.append((fpr, tpr))
        
        # Calcula el área bajo la curva ROC (AUC)
        auc = roc_auc_score(y_test, y_pred_prob)
        roc_aucs.append(auc)
    
    
    return score_best,score_worse,score_mean,roc_curves,roc_aucs


primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
n = 145
primos_dict = [{'n_neighbors': i} for i in primos]
score_best,score_worse,score_mean,roc_curves,roc_aucs = evaluate_model(KNeighborsClassifier, x_train,x_test,
                                                                       y_train,y_test,
                                                                       primos_dict, cv=5)

    
# Gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de puntajes de validación cruzada
ax1.plot(primos, score_best, "o--", label="Mejor score obtenido")
ax1.plot(primos, score_worse, "o--", label="Peor score obtenido")
ax1.plot(primos, score_mean, "o--", label="Promedio score obtenido")
ax1.axhline(mejor_score,linestyle = ":", color = "green" ,label = "Mejor score medio obtenido en CV")
ax1.axvline(mejor_param,linestyle = ":", color = "green" ,label = "Mejor parametro encontrado en CV")
ax1.legend()
ax1.set_xlabel("Hiperparámetros")
ax1.set_ylabel("Performance")
ax1.set_title("Puntajes de CrossValidation del modelo\n para diferentes hiperparámetros")
ax1.grid()

# Gráfico de curvas ROC
for i,param  in enumerate(fibonacci):
    fpr, tpr = roc_curves[i]
    ax2.plot(fpr, tpr, label=f" {param}, AUC={roc_aucs[i]:.3f}")

ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.set_title('Curvas ROC')
ax2.legend(title = "Cantitad de vecinos y AUC")
ax2.grid()
plt.tight_layout()

plt.show()
## se puede observar que el mejor parametro (34) y los parametros de 2 a 89
#son muy similares en rendimiento maximo y en la curva roc, en especial queda
#destacar que tanto n_neighbors = 5 y n_neighbors = 34 tienen putajes muy similares
#tanto en score promedio y score minimo tambien, entonces si bien 34 puede ser
#el mejor, con n_neighbors = 5 consigo un trabajo similar y con menores calculos
