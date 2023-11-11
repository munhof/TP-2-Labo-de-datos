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
#print(tree_distancia.score(df_dist_test,y_test))
#plot_tree(tree_distancia)


#dot_data = export_graphviz(tree_distancia, out_file=None, 
                                    # feature_names= df_dist_train.columns,
                                    # class_names= ["remera",
                                    #      "pantalon",
                                    #      "pullover",
                                    #      "vestidos",
                                    #      "camperas",
                                    #      "sandalias",
                                    #      "camisetas",
                                    #      "zapatillas",
                                    #      "bolsos",
                                    #      "botas"],
                                    # filled=True, rounded=True,
                                    # special_characters=True)
#graph = graphviz.Source(dot_data) #Armamos el grafo
#graph.render("titanic", format= "png") #Guardar la imágen

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

#buscamos los mejores parametro


hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [3,5,7,10,11,20,50,80,110],
                }

#defino el modelo
tree_distancia = DecisionTreeClassifier(random_state = 5) 

clf = GridSearchCV(tree_distancia, hyper_params,cv = 5)
search = clf.fit(df_dist_train,y_train)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param_altura = search.best_params_["max_depth"]
mejor_param_criterio = search.best_params_["criterion"]
mejor_score = search.best_score_
print(f"mejor score {search.best_score_}")
#lo graficamos
# optimo_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11,
#                                      random_state = 5) 
# optimo_tree.fit(df_dist_train,y_train)
# dot_data = export_graphviz(optimo_tree, out_file=None, 
#                                     feature_names= df_dist_train.columns,
#                                     class_names= ["remera",
#                                          "pantalon",
#                                          "pullover",
#                                          "vestidos",
#                                          "camperas",
#                                          "sandalias",
#                                          "camisetas",
#                                          "zapatillas",
#                                          "bolsos",
#                                          "botas"],
#                                     filled=True, rounded=True,
#                                     special_characters=True)
#graph = graphviz.Source(dot_data) #Armamos el grafo
#graph.render("titanic", format= "png") #Guardar la imágen


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

score_best, score_worse, score_mean, classification_reports = evaluate_model(model = DecisionTreeClassifier,
                                                                       X_train = df_dist_train, 
                                                                       X_test = df_dist_test,
                                                                       y_train = y_train,
                                                                       y_test = y_test,
                                                                       hyperparameters = hyper_params, 
                                                                       cv = 5)

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

# Función para imprimir un informe de clasificación con formato personalizado
def print_classification_report_custom(report, class_names):
    print("Classification report rebuilt from confusion matrix:")
    header = ["precision", "recall", "f1-score", "support"]
    line = "{:<10} " + "{:<10} " * (len(header) - 1)
    print(line.format("", *header))

    for i, class_name in enumerate(class_names):
        line_data = [report[class_name][metric] for metric in header]
        print(line.format(class_name, *line_data))

    accuracy = report["accuracy"]
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    line_data = [accuracy, macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"], macro_avg["support"]]
    print(line.format("accuracy", *line_data))

    line_data = [weighted_avg["precision"], weighted_avg["recall"], weighted_avg["f1-score"], weighted_avg["support"]]
    print(line.format("weighted avg", *line_data))

# Imprimir los informes de clasificación para el peor caso, el mejor caso y un caso medio en gini
print("Informe de clasificación (gini) - Peor Caso")
print_classification_report_custom(gini_reports[worst_gini_idx], class_names)
print("\nInforme de clasificación (gini) - Mejor Caso")
print_classification_report_custom(gini_reports[best_gini_idx], class_names)
print("\nInforme de clasificación (gini) - Caso Medio")
print_classification_report_custom(gini_reports[mean_gini_idx], class_names)

# Imprimir los informes de clasificación para el peor caso, el mejor caso y un caso medio en entropy
print("Informe de clasificación (entropy) - Peor Caso")
print_classification_report_custom(entropy_reports[worst_entropy_idx], class_names)
print("\nInforme de clasificación (entropy) - Mejor Caso")
print_classification_report_custom(entropy_reports[best_entropy_idx], class_names)
print("\nInforme de clasificación (entropy) - Caso Medio")
print_classification_report_custom(entropy_reports[mean_entropy_idx], class_names)


def save_classification_report_to_csv(report, class_names, filename):
    data = {
        'Class': class_names,
        'Precision': [report[class_name]['precision'] for class_name in class_names],
        'Recall': [report[class_name]['recall'] for class_name in class_names],
        'F1-Score': [report[class_name]['f1-score'] for class_name in class_names],
        'Support': [report[class_name]['support'] for class_name in class_names]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
# Guardar los informes de clasificación en archivos CSV
save_classification_report_to_csv(gini_reports[worst_gini_idx], class_names, "./reportes_clasificacion/worst_gini_report.csv")
save_classification_report_to_csv(gini_reports[best_gini_idx], class_names, "./reportes_clasificacion/best_gini_report.csv")
save_classification_report_to_csv(gini_reports[mean_gini_idx], class_names, "./reportes_clasificacion/mean_gini_report.csv")

save_classification_report_to_csv(entropy_reports[worst_entropy_idx], class_names, "./reportes_clasificacion/worst_entropy_report.csv")
save_classification_report_to_csv(entropy_reports[best_entropy_idx], class_names, "./reportes_clasificacion/best_entropy_report.csv")
save_classification_report_to_csv(entropy_reports[mean_entropy_idx], class_names, "./reportes_clasificacion/mean_entropy_report.csv")
