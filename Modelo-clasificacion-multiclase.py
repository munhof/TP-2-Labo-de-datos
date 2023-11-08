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
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import graphviz

import Limpieza_de_datos

#en este codigo vamos a probar varios metodos de ajuste con knn con el cual
#vamos a evaluar la implementacion de varios algoritmos

#%% Distancia a los arquetipos
#unos de los metodos es la clasificacion mediante la distancia a los 
#arquetipos
X_train,X_test,y_train,y_test = Limpieza_de_datos.data_train_modelo_multiclase()
arquetipos = Limpieza_de_datos.prendas_arquetipicas()

def distancia_arquetipo(data_frame):
    global arquetipos
    distancias = []
    n = np.shape(data_frame)[0]
    for j in range(10):
        arquetipo = arquetipos[j]
        distancia_prenda=[]
        for i in range(n):
            data_prenda = data_frame.iloc[i]
            array_prenda = data_prenda.to_numpy()
            array_arquetipo = arquetipo.to_numpy()
            distancia_prenda.append(np.linalg.norm(array_prenda/255-array_arquetipo/255))
        distancias.append(distancia_prenda)
    
    return distancias

distancias_train =  np.array(distancia_arquetipo(X_train))
distancias_test =  np.array(distancia_arquetipo(X_test))

keys = ["remera",
        "pantalon",
        "pullover",
        "vestidos",
        "camperas",
        "sandalias",
        "camisetas",
        "zapatillas",
        "bolsos",
        "botas"]

df_dist_train = pd.DataFrame({})
df_dist_test = pd.DataFrame({})

for i, key in enumerate(keys):
    df_dist_test[key] = distancias_test[i]
    df_dist_train[key] = distancias_train[i]

#armo el modelo, arbitrariamente elijo profundidad 4, criterio gini
tree_distancia = DecisionTreeClassifier(criterion='gini',random_state= 5,
                                        max_depth= 4,) 

#entreno el modelo
tree_distancia.fit(df_dist_train,y_train)

#vemos el escore 
print(tree_distancia.score(df_dist_test,y_test))
plot_tree(tree_distancia)


dot_data = export_graphviz(tree_distancia, out_file=None, 
                                    feature_names= df_dist_train.columns,
                                    class_names= ["remera",
                                         "pantalon",
                                         "pullover",
                                         "vestidos",
                                         "camperas",
                                         "sandalias",
                                         "camisetas",
                                         "zapatillas",
                                         "bolsos",
                                         "botas"],
                                    filled=True, rounded=True,
                                    special_characters=True)
graph = graphviz.Source(dot_data) #Armamos el grafo
graph.render("titanic", format= "png") #Guardar la imágen

data_original = df_dist_test.copy()
data_original["label"] = y_test.copy()

data_predict = df_dist_test.copy()
data_predict["label"] = tree_distancia.predict(df_dist_test)


#plt.plot()
#sns.pairplot(data_original, hue="label")
#plt.show()

#plt.plot()
#sns.pairplot(data_predict, hue="label")
#plt.show()

#buscamos los mejores parametros

data_X  = pd.concat([df_dist_train,df_dist_test])
data_y = pd.concat([y_train,y_test])


hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [3,5,7,10,11,20,50,80,110],
                }

#defino el modelo
tree_distancia = DecisionTreeClassifier(random_state = 5) 

clf = GridSearchCV(tree_distancia, hyper_params,cv = 20)
search = clf.fit(data_X,data_y)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
print(f"mejor score {search.best_score_}")
#lo graficamos
optimo_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11,
                                     random_state = 5) 
optimo_tree.fit(df_dist_train,y_train)
dot_data = export_graphviz(optimo_tree, out_file=None, 
                                    feature_names= df_dist_train.columns,
                                    class_names= ["remera",
                                         "pantalon",
                                         "pullover",
                                         "vestidos",
                                         "camperas",
                                         "sandalias",
                                         "camisetas",
                                         "zapatillas",
                                         "bolsos",
                                         "botas"],
                                    filled=True, rounded=True,
                                    special_characters=True)
graph = graphviz.Source(dot_data) #Armamos el grafo
graph.render("titanic", format= "png") #Guardar la imágen


#hay entre un arbol de 11 y 5 de altura hay una diferencia de 0.01 en
#score, verifico las curvas roc y la evolucion de rendimiento

def evaluate_model(model, X, y, hyperparameters, cv):
    # Listas para almacenar los resultados
    score_best = []
    score_worse = []
    score_mean = []
    roc_curves = []
    roc_aucs = []

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=3)

    for param in hyperparameters:
        clf = model(**param)
        
        # Calcula los puntajes de validación cruzada
        score_cv = cross_val_score(clf, X, y, cv=kf)
        score_best.append(max(score_cv))
        score_worse.append(min(score_cv))
        score_mean.append(score_cv.mean())
        
        # Listas para almacenar las predicciones y etiquetas reales
        y_pred_all = []
        y_true_all = []
        
        for train_idx, test_idx in list(kf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Ajusta el modelo
            clf.fit(X_train, y_train)
        
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
        
            y_pred_all = y_pred_all + list(y_pred_prob)
            y_true_all = y_true_all + list(y_test.values.reshape(np.shape(y_test)[0]))
            
        # Después de completar todos los folds
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
        
        # Calcula la curva ROC
        roc_curves.append((fpr, tpr))
        
        # Calcula el área bajo la curva ROC (AUC)
        auc = roc_auc_score(y_true_all, y_pred_all)
        roc_aucs.append(auc)
    
    
    return score_best,score_worse,score_mean,roc_curves,roc_aucs
