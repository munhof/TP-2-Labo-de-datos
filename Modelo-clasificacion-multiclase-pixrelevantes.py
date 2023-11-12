#%% -*- coding: utf-8 -*-
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
    
Modelo arboles de decision para clasificacion multiclase
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
#import graphviz

import Limpieza_de_datos

#en este codigo vamos a probar varios metodos de ajuste con knn con el cual
#vamos a evaluar la implementacion de varios algoritmos

# Modelo pixeles relevantes
#%% A cada prenda arquetipica le resto el promedio de todas las prendas arquetipicas, sin contar esta. Los pixeles de mayor valor
# me muestran las zonas mas representativas de la prenda arquetipica, mientras que los de menor valor muestran las menos representativas.

def regiones_de_interes(n):
    arquetipos = Limpieza_de_datos.prendas_arquetipicas()
    media = np.zeros((1,784))
    for i in range(10):
        if i != n:
            media += arquetipos[i].to_numpy()
    media_promedio = media/9
    return media_promedio

prendas_menos_media = []
for i in range(10):
    prendas_menos_media.append(Limpieza_de_datos.prendas_arquetipicas()[i].to_numpy()-regiones_de_interes(i))

#%% Elijo los 6-7 valores maximos y 6-7 valores minimos de cada prenda arquetipica

max_y_min = []
for i in range(10):
    max_y_min.append((prendas_menos_media[i].max(),prendas_menos_media[i].min()))

pixeles_de_interes = {}
pixeles_de_interes[0] = (np.where(prendas_menos_media[0]>100)[1],np.where(prendas_menos_media[0]<-95)[1])
pixeles_de_interes[1] = (np.where(prendas_menos_media[1]>104)[1],np.where(prendas_menos_media[1]<-127)[1])
pixeles_de_interes[2] = (np.where(prendas_menos_media[2]>107)[1],np.where(prendas_menos_media[2]<-53)[1])
pixeles_de_interes[3] = (np.where(prendas_menos_media[3]>104)[1],np.where(prendas_menos_media[3]<-100)[1])
pixeles_de_interes[4] = (np.where(prendas_menos_media[4]>107)[1],np.where(prendas_menos_media[4]<-59)[1])
pixeles_de_interes[5] = (np.where(prendas_menos_media[5]>54)[1],np.where(prendas_menos_media[5]<-119)[1])
pixeles_de_interes[6] = (np.where(prendas_menos_media[6]>78)[1],np.where(prendas_menos_media[6]<-52)[1])
pixeles_de_interes[7] = (np.where(prendas_menos_media[7]>124)[1],np.where(prendas_menos_media[7]<-132)[1])
pixeles_de_interes[8] = (np.where(prendas_menos_media[8]>93)[1],np.where(prendas_menos_media[8]<-85)[1])
pixeles_de_interes[9] = (np.where(prendas_menos_media[9]>138)[1],np.where(prendas_menos_media[9]<-118)[1])

#%% defino la funcion que me transformador, esta recibe un dataframe y devuelve el dataframe con solo los nuevos atributos ["Es remera", "Es pantalon",...]
# estos nuevos atributos tienen un valor num que es la suma de los pixeles relevantes menos la suma de los no relevantes.
def transformador(data):
    data["pixeles_remeras"] =       data[['pixel146', 'pixel737', 'pixel738', 'pixel743', 'pixel748', 'pixel749']].sum(axis=1)
    data["pixeles_no_remeras"] =    data[['pixel415', 'pixel442', 'pixel443', 'pixel470', 'pixel471', 'pixel499', 'pixel527']].sum(axis=1)
    data["pixeles_pantalon"] =      data[['pixel39', 'pixel40', 'pixel42', 'pixel43', 'pixel46', 'pixel67', 'pixel95']].sum(axis=1)
    data["pixeles_no_pantalon"] =   data[['pixel386', 'pixel435', 'pixel442', 'pixel463', 'pixel491', 'pixel519', 'pixel547', 'pixel575']].sum(axis=1)
    data["pixeles_pullover"] =      data[['pixel287', 'pixel315', 'pixel343', 'pixel371', 'pixel734', 'pixel751']].sum(axis=1)
    data["pixeles_no_pullover"] =   data[['pixel334', 'pixel362', 'pixel390', 'pixel419', 'pixel447', 'pixel475']].sum(axis=1)
    data["pixeles_vestidos"] =      data[['pixel715', 'pixel741', 'pixel742', 'pixel743', 'pixel744', 'pixel745']].sum(axis=1)
    data["pixeles_no_vestidos"] =   data[['pixel387', 'pixel415', 'pixel443', 'pixel444', 'pixel471', 'pixel472', 'pixel499']].sum(axis=1)
    data["pixeles_camperas"] =      data[['pixel288', 'pixel289', 'pixel316', 'pixel317', 'pixel344', 'pixel372']].sum(axis=1)
    data["pixeles_no_camperas"] =   data[['pixel334', 'pixel362', 'pixel390', 'pixel418', 'pixel446', 'pixel474', 'pixel475']].sum(axis=1)
    data["pixeles_sandalias"] =     data[['pixel474', 'pixel475', 'pixel502', 'pixel503', 'pixel530']].sum(axis=1)
    data["pixeles_no_sandalias"] =  data[['pixel41',  'pixel42',  'pixel44',  'pixel45', 'pixel606', 'pixel633', 'pixel634']].sum(axis=1)
    data["pixeles_camisetas"] =     data[['pixel93', 'pixel120', 'pixel232', 'pixel260', 'pixel288', 'pixel316']].sum(axis=1)
    data["pixeles_no_camisetas"] =  data[['pixel390', 'pixel418', 'pixel446', 'pixel447', 'pixel474', 'pixel475']].sum(axis=1)
    data["pixeles_zapatillas"] =    data[['pixel362', 'pixel390', 'pixel391', 'pixel418', 'pixel419', 'pixel447']].sum(axis=1)
    data["pixeles_no_zapatillas"] = data[['pixel600', 'pixel627', 'pixel628', 'pixel629', 'pixel656', 'pixel657']].sum(axis=1)
    data["pixeles_bolsos"] =        data[['pixel340', 'pixel341', 'pixel368', 'pixel369', 'pixel370', 'pixel396', 'pixel397']].sum(axis=1)
    data["pixeles_no_bolsos"] =     data[['pixel40',  'pixel70',  'pixel71',  'pixel98',  'pixel99',  'pixel100', 'pixel127']].sum(axis=1)
    data["pixeles_botas"] =         data[['pixel249', 'pixel277', 'pixel530', 'pixel558', 'pixel559', 'pixel586']].sum(axis=1)
    data["pixeles_no_botas"] =      data[['pixel41', 'pixel291', 'pixel319', 'pixel346', 'pixel347', 'pixel374', 'pixel375']].sum(axis=1)
    data["Es remera"] = data["pixeles_remeras"]-data["pixeles_no_remeras"]
    data["Es pantalon"] = data["pixeles_pantalon"]-data["pixeles_no_pantalon"]
    data["Es pullover"] = data["pixeles_pullover"]-data["pixeles_no_pullover"]
    data["Es vestido"] = data["pixeles_vestidos"]-data["pixeles_no_vestidos"]
    data["Es campera"] = data["pixeles_camperas"]-data["pixeles_no_camperas"]
    data["Es sandalia"] = data["pixeles_sandalias"]-data["pixeles_no_sandalias"]
    data["Es camiseta"] = data["pixeles_camisetas"]-data["pixeles_no_camisetas"]
    data["Es zapatilla"] = data["pixeles_zapatillas"]-data["pixeles_no_zapatillas"]
    data["Es bolso"] = data["pixeles_bolsos"]-data["pixeles_no_bolsos"]
    data["Es bota"] = data["pixeles_botas"]-data["pixeles_no_botas"]
    data = pd.DataFrame(data[["Es remera", "Es pantalon",
                              "Es pullover","Es vestido",
                              "Es campera","Es sandalia",
                              "Es camiseta","Es zapatilla",
                              "Es bolso","Es bota"]])
    return data


#%% 
# separo el dataset
X_train,X_test,y_train,y_test = Limpieza_de_datos.data_train_modelo_multiclase()

x_train = transformador(X_train)
x_test = transformador(X_test)

#buscamos los mejores parametro


hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [3,5,7,10,11,20,50,80,110],
                }

#defino el modelo
tree_distancia = DecisionTreeClassifier(random_state = 5) 

clf = GridSearchCV(tree_distancia, hyper_params,cv = 5)
search = clf.fit(x_train,y_train)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param_altura = search.best_params_["max_depth"]
mejor_param_criterio = search.best_params_["criterion"]
mejor_score = search.best_score_
print(f"mejor score {search.best_score_}")

#hay entre un arbol de 11 y 5 de altura hay una diferencia de 0.01 en
#score, verifico las curvas roc y la evolucion de rendimiento

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pandas as pd

def evaluate_model(model, X_train, X_test, y_train, y_test, hyperparameters, cv):
    # Listas para almacenar los resultados
    score_best = []
    score_worse = []
    score_mean = []
    classification_reports = []
    
    class_names = [
        "remera",
        "pantalon",
        "pullover",
        "vestidos",
        "camperas",
        "sandalias",
        "camisetas",
        "zapatillas",
        "bolsos",
        "botas"
    ]
    
    for criterio in hyperparameters["criterion"]:
        for max_profundidad in hyperparameters["max_depth"]:
            params = {"criterion": criterio, "max_depth": max_profundidad, "random_state": 5}
            clf = model(**params)
            
            # Calcula los puntajes de validación cruzada
            score_cv = cross_val_score(clf, X_train, y_train, cv=cv)
            
            score_best.append((criterio, max_profundidad, max(score_cv)))
            score_worse.append((criterio, max_profundidad, min(score_cv)))
            score_mean.append((criterio, max_profundidad, score_cv.mean()))
            
            # Entrena el clasificador en el conjunto de entrenamiento
            clf.fit(X_train, y_train)
            
            # Realiza predicciones en el conjunto de prueba
            y_pred = clf.predict(X_test)
            
            # Genera el informe de clasificación
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            classification_reports.append(report)
    
    score_best = pd.DataFrame(score_best, columns=["criterion", "max_depth", "score"])
    score_worse = pd.DataFrame(score_worse, columns=["criterion", "max_depth", "score"])
    score_mean = pd.DataFrame(score_mean, columns=["criterion", "max_depth", "score"])
    
    return score_best, score_worse, score_mean, classification_reports

hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [5,7,10,11,20,50],
                }

score_best, score_worse, score_mean, classification_reports = evaluate_model(DecisionTreeClassifier, x_train, x_test, y_train,y_test, hyper_params, 5)

class_names= ["remera",
              "pantalon",
              "pullover",
              "vestidos",
              "camperas",
              "sandalias",
              "camisetas",
              "zapatillas",
              "bolsos",
              "botas"]


# Gráfico de puntajes de validación cruzada para gini y entropy
fig, ax1 = plt.subplots(figsize=(10, 5))

# Filtrar los datos por criterio
gini_mask = score_best["criterion"] == "gini"
entropy_mask = score_best["criterion"] == "entropy"

# Gráfico para gini
ax1.plot(
    score_best[gini_mask]["max_depth"],
    score_best[gini_mask]["score"],
    "ro-",
    label="Mejor score (gini)",
)
ax1.plot(
    score_worse[gini_mask]["max_depth"],
    score_worse[gini_mask]["score"],
    "go-",
    label="Peor score (gini)",
)
ax1.plot(
    score_mean[gini_mask]["max_depth"],
    score_mean[gini_mask]["score"],
    "bo-",
    label="Promedio score (gini)",
)

# Gráfico para entropy
ax1.plot(
    score_best[entropy_mask]["max_depth"],
    score_best[entropy_mask]["score"],
    "ro--",
    label="Mejor score (entropy)",
)
ax1.plot(
    score_worse[entropy_mask]["max_depth"],
    score_worse[entropy_mask]["score"],
    "go--",
    label="Peor score (entropy)",
)
ax1.plot(
    score_mean[entropy_mask]["max_depth"],
    score_mean[entropy_mask]["score"],
    "bo--",
    label="Promedio score (entropy)",
)

ax1.axhline(mejor_score, linestyle=":", color="green", label=f"Mejor score medio obtenido en CV\n criterio:{mejor_param_criterio }")
ax1.axvline(mejor_param_altura, linestyle=":", color="green", label=f"Mejor score medio obtenido en CV\n criterio:{mejor_param_criterio }")
ax1.legend()
ax1.set_xlabel("Profundidad máxima del árbol")
ax1.set_ylabel("Performance")
ax1.set_title("Puntajes de CrossValidation del modelo")
ax1.grid()

plt.show()

# Filtrar los informes por criterio
gini_reports = [report for i, report in enumerate(classification_reports) if gini_mask[i]]
entropy_reports = [report for i, report in enumerate(classification_reports) if entropy_mask[i]]

# %%
