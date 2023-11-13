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


#%%
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
#%%
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
plt.savefig('.\modelo-multiclase\Clasificacion-distancia-evaluacion.png')
plt.show()
#%%
# Filtrar los informes por criterio
gini_reports = [report for i, report in enumerate(classification_reports) if gini_mask[i]]
entropy_reports = [report for i, report in enumerate(classification_reports) if entropy_mask[i]]
worst_gini_idx = np.argmin(score_worse[gini_mask]["score"])
best_gini_idx = np.argmax(score_best[gini_mask]["score"])
mean_gini_idx = np.argmax(score_mean[gini_mask]["score"])

worst_entropy_idx = np.argmin(score_worse[entropy_mask]["score"])
best_entropy_idx = np.argmax(score_best[entropy_mask]["score"])
mean_entropy_idx = np.argmax(score_mean[entropy_mask]["score"])
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

#%% aca arranca el modelo multiclase de correlacion

from Limpieza_de_datos import data_train_modelo_pantalon_remera, data_train_modelo_multiclase
from sklearn import linear_model
from sklearn.metrics import r2_score, classification_report, roc_curve, roc_auc_score,precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz

X_train, X_test, y_train, y_test = data_train_modelo_pantalon_remera(  )


pantalones_y_remeras = X_train


idex_de_pixeles_candidato = list(pantalones_y_remeras.std().nlargest(
    n=30).index)  # Me da los 30 pixeles con mayor
# std entre las dos clases en forma de lista, los llamo candidato porque pasaron el filtro del std.

# Tengo que calcular correlacion, necesito
pixeles_candidato = pantalones_y_remeras[idex_de_pixeles_candidato]
# la informacion de la iluminacion de los candidatos en cada imagen.
del pantalones_y_remeras


# Dado el scatterplot de un pixel contra otro estos no van a estar correlacionados si dicho grafico esta homogeneamente
# cubierto de puntos, si estan concentrados en un cono entonces lo estan, la idea es hacer un ajuste lineal y calcular
# el error del ajuste, quiero que el error sea muy grande porque eso quiere decir que el ajuste no es significativo,
# entonces no se puede calcular la iluminacion de un pixel agarrando otro y haciendo una cuenta.
# Uso error y correlacion indistintamente pues cuando el error es alto la correlacion es baja.
# Primero hago esto para todo par de pixeles de la lista de pixeles candidato, luego expando a 3.

def correlacion_entre_pixeles(pixeles_candidato):
    columnas = pixeles_candidato.columns
    cant_candidatos = len(columnas)
    pixel1 = []
    pixel2 = []
    # error[k] es el error del ajuste lineal entre el pixel1[k] y pixel2[k].
    error = []
    for i in range(cant_candidatos - 1):
        primer_pixel = columnas[i]
        info_primer_pixel = pixeles_candidato[[primer_pixel]]
        for j in range(i+1, cant_candidatos):
            segundo_pixel = columnas[j]
            info_segundo_pixel = pixeles_candidato[[segundo_pixel]]
            rls = linear_model.LinearRegression()
            rls.fit(info_primer_pixel, info_segundo_pixel)
            # Hice el ajuste lineal, luego falta el error de dicho ajuste.
            pixel1.append(primer_pixel)
            pixel2.append(segundo_pixel)
            su_error = error_lineal_medio(ajuste_lineal, rls, primer_pixel, segundo_pixel,
                                          info_primer_pixel, info_segundo_pixel)

            error.append(float(su_error))
    tabla_de_correlacion = pd.DataFrame({'Primer_Pixel':pixel1, 
                                           'Segundo_Pixel':pixel2, 
                                           'Su_error': error})
    # En cada fila de esta tabla hay dos pixeles y su correlacion, sin repetir.
    return tabla_de_correlacion

def ajuste_lineal(rls, x):    # La recta que salio del ajuste lineal.
    a = rls.coef_[0][0]
    b = rls.intercept_[0]
    y = a*x + b
    return y

def error_lineal_medio(ajuste_lineal, rls, primer_pixel, segundo_pixel,
                       info_primer_pixel, info_segundo_pixel):
    res = 0
    for i, j in zip(info_primer_pixel[primer_pixel], info_segundo_pixel[segundo_pixel]):
        res = res + np.linalg.norm(ajuste_lineal(rls, i) - j)
        # Igual al error cuadratico medio pero sin la raiz ni el cuadrado, solo el modulo: Dado un punto
        # calculo la distancia a la recta solo en el eje y, sumo al resultado y repito con todos los puntos.
        # Solo me quedo con el promedio, sino queda un numero demasiado grande.
    res = res/(info_primer_pixel.shape[0])
    return res


tabla_correlaciones = correlacion_entre_pixeles(pixeles_candidato) 
del pixeles_candidato # Si agarrara el maximo de esta tabla tendria el par de 
# pixeles con menor correlacion, pero la idea es llegar a 3.
# Ya que tengo la correlacion para todo par de pixeles de los candidatos, dado un pixel i y uno j de 
# los candidato, puedo agarrar la correlacion entre i y j, la correlacion de i con un k de los candidato y la de
# j con k, sumarlos y tener una medida para la correlacion del trio, asi que voy a hacer eso para todo i, j y k
# de los candidatos.

from inline_sql import sql, sql_val

# Primero voy a generar la tabla de 3 columnas donde estan todas las convinaciones de los 30 pixeles candidato.
# Para eso agarro los 30 y procedo a hacer una serie de inner joins.
mono_pixel = pd.DataFrame()
mono_pixel['Primer_pixel'] = pd.DataFrame(idex_de_pixeles_candidato)
del idex_de_pixeles_candidato

consultaSQL1 ="""
SELECT mono1.Primer_pixel, mono2.Primer_pixel AS Segundo_pixel
FROM mono_pixel AS mono1
INNER JOIN mono_pixel mono2
ON mono1.Primer_pixel != mono2.Primer_pixel
"""

# Esta es la tabla con todas las convinaciones de 2 pixeles de los 30 teniendo cuidado de que en ninguna fila
# se repitan columnas.
duo_pixel = sql^consultaSQL1
del mono_pixel
del consultaSQL1

consultaSQL2 ="""
SELECT duo1.Primer_pixel, duo1.Segundo_pixel, duo2.Segundo_pixel AS Tercer_pixel
FROM duo_pixel AS duo1
INNER JOIN duo_pixel AS duo2
ON duo1.Segundo_pixel = duo2.Primer_pixel AND duo1.Primer_pixel != duo2.Segundo_pixel
"""

# Esta es la tabla con todas las convinaciones de 3 pixeles de los 30 teniendo cuidado de que en ninguna fila
# se repitan columnas.
trio_pixel = sql^consultaSQL2
del duo_pixel
del consultaSQL2

consultaSQL3 ="""
SELECT tp.Primer_pixel, tp.Segundo_pixel, tp.Tercer_Pixel, Su_error AS Su_error_1_2
FROM trio_pixel AS tp
INNER JOIN tabla_correlaciones AS TC
ON tp.Primer_pixel = TC.Primer_pixel AND tp.Segundo_pixel = TC.Segundo_pixel OR tp.Segundo_pixel = TC.Primer_pixel AND tp.Primer_pixel = TC.Segundo_pixel
"""

# A la anterior tabla le anexo el error entre el primer pixel y el segundo.
primer_error = sql^consultaSQL3
del consultaSQL3

consultaSQL4 ="""
SELECT pe.Primer_pixel, pe.Segundo_pixel, pe.Tercer_Pixel, pe.Su_error_1_2, TC.Su_error AS Su_error_1_3
FROM primer_error AS pe
INNER JOIN tabla_correlaciones AS TC
ON pe.Primer_pixel = TC.Primer_pixel AND pe.Tercer_pixel = TC.Segundo_pixel OR pe.Tercer_pixel = TC.Primer_pixel AND pe.Primer_pixel = TC.Segundo_pixel
"""

# A la anterior tabla le anexo el error entre el primer pixel y el tercero.
segundo_error = sql^consultaSQL4
del primer_error
del consultaSQL4

consultaSQL5 ="""
SELECT se.Primer_pixel, se.Segundo_pixel, se.Tercer_Pixel, se.Su_error_1_2, Su_error_1_3, TC.Su_error AS Su_error_2_3
FROM segundo_error AS se
INNER JOIN tabla_correlaciones AS TC
ON se.Segundo_pixel = TC.Primer_pixel AND se.Tercer_pixel = TC.Segundo_pixel OR se.Tercer_pixel = TC.Primer_pixel AND se.Segundo_pixel = TC.Segundo_pixel
"""

# A la anterior tabla le anexo el error entre el segundo pixel y el tercero.
tercer_error = sql^consultaSQL5
del segundo_error
del trio_pixel
del consultaSQL5
del tabla_correlaciones

consultaSQL6 = """
SELECT Primer_pixel, Segundo_pixel, Tercer_pixel, (Su_error_1_2 + Su_error_1_3 + Su_error_2_3) AS Correlacion
FROM tercer_error
"""

# En la anterior tabla reemplazo las columnas de los errores por su suma, que es lo que me interesaba.
correlacion_trio_pixeles = sql^consultaSQL6
del consultaSQL6
del tercer_error

# Me quedo con la el error mas alto, pues me da la correlacion mas baja.
maximo = correlacion_trio_pixeles['Correlacion'].max()
# Tuve que usar iloc pues tener cuidado con los inner joins no me salvo de que tuviera filas que estan de mas
# donde simplemente se habian permutado las columnas, pero eso no me afecta, simplemente calculo el maximo y 
# de las 5 filas me quedo con 1 fila, no elimino ñas filas de mas porque solo me interesa el maximo.
fila_del_maximo = (correlacion_trio_pixeles[correlacion_trio_pixeles['Correlacion'] == maximo]).iloc[0]
# Me quedo solo con los pixeles.
pixeles_elejidos = fila_del_maximo[['Primer_pixel','Segundo_pixel','Tercer_pixel']]
del fila_del_maximo
del correlacion_trio_pixeles
del maximo


# Entreno y testeo un modelo de arbol seleccionando los pixeles con el metodo visto anteriormente con el dataset entero.

X_train,X_test,y_train,y_test = data_train_modelo_multiclase()

x_train = X_train[pixeles_elejidos]
x_test = X_test[pixeles_elejidos]

hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [3,5,7,10,11,20,50,80,110],
                }

#defino el modelo
tree_correlacion = DecisionTreeClassifier(random_state = 5) 

clf = GridSearchCV(tree_correlacion, hyper_params,cv = 5)
search = clf.fit(x_train,y_train)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param_altura = search.best_params_["max_depth"]
mejor_param_criterio = search.best_params_["criterion"]
mejor_score = search.best_score_
print(f"mejor score {search.best_score_}")


hyper_params = {"criterion": ["gini", "entropy"],
                "max_depth": [5,7,10,11,20,50],
                }

score_best, score_worse, score_mean, classification_reports = evaluate_model(model = DecisionTreeClassifier,
                                                                       X_train = x_train, 
                                                                       X_test = x_test,
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
plt.savefig('.\modelo-multiclase\Clasificacion-correlacion-evaluacion.png')

plt.show()
#%%
# Filtrar los informes por criterio
gini_reports = [report for i, report in enumerate(classification_reports) if gini_mask[i]]
entropy_reports = [report for i, report in enumerate(classification_reports) if entropy_mask[i]]

worst_gini_idx = np.argmin(score_worse[gini_mask]["score"])
best_gini_idx = np.argmax(score_best[gini_mask]["score"])
mean_gini_idx = np.argmax(score_mean[gini_mask]["score"])

worst_entropy_idx = np.argmin(score_worse[entropy_mask]["score"])
best_entropy_idx = np.argmax(score_best[entropy_mask]["score"])
mean_entropy_idx = np.argmax(score_mean[entropy_mask]["score"])

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
save_classification_report_to_csv(gini_reports[worst_gini_idx], class_names, "./reportes_clasificacion/worst_gini_report_correlacion.csv")
save_classification_report_to_csv(gini_reports[best_gini_idx], class_names, "./reportes_clasificacion/best_gini_report_correlacion.csv")
save_classification_report_to_csv(gini_reports[mean_gini_idx], class_names, "./reportes_clasificacion/mean_gini_report_correlacion.csv")

save_classification_report_to_csv(entropy_reports[worst_entropy_idx], class_names, "./reportes_clasificacion/worst_entropy_report_correlacion.csv")
save_classification_report_to_csv(entropy_reports[best_entropy_idx], class_names, "./reportes_clasificacion/best_entropy_report_correlacion.csv")
save_classification_report_to_csv(entropy_reports[mean_entropy_idx], class_names, "./reportes_clasificacion/mean_entropy_report_correlacion.csv")
#%% aca arranca el modelo por regiones
#A cada prenda arquetipica le resto el promedio de todas las prendas arquetipicas, sin contar esta. Los pixeles de mayor valor
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
tree_regiones = DecisionTreeClassifier(random_state = 5) 

clf = GridSearchCV(tree_regiones, hyper_params,cv = 5)
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
plt.savefig('.\modelo-multiclase\Clasificacion-regiones-evaluacion.png')

plt.show()
# Filtrar los informes por criterio
gini_reports = [report for i, report in enumerate(classification_reports) if gini_mask[i]]
entropy_reports = [report for i, report in enumerate(classification_reports) if entropy_mask[i]]
worst_gini_idx = np.argmin(score_worse[gini_mask]["score"])
best_gini_idx = np.argmax(score_best[gini_mask]["score"])
mean_gini_idx = np.argmax(score_mean[gini_mask]["score"])

worst_entropy_idx = np.argmin(score_worse[entropy_mask]["score"])
best_entropy_idx = np.argmax(score_best[entropy_mask]["score"])
mean_entropy_idx = np.argmax(score_mean[entropy_mask]["score"])

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
save_classification_report_to_csv(gini_reports[worst_gini_idx], class_names, "./reportes_clasificacion/worst_gini_report_regiones.csv")
save_classification_report_to_csv(gini_reports[best_gini_idx], class_names, "./reportes_clasificacion/best_gini_report_regiones.csv")
save_classification_report_to_csv(gini_reports[mean_gini_idx], class_names, "./reportes_clasificacion/mean_gini_report_regiones.csv")

save_classification_report_to_csv(entropy_reports[worst_entropy_idx], class_names, "./reportes_clasificacion/worst_entropy_report_regionescsv")
save_classification_report_to_csv(entropy_reports[best_entropy_idx], class_names, "./reportes_clasificacion/best_entropy_report_regiones.csv")
save_classification_report_to_csv(entropy_reports[mean_entropy_idx], class_names, "./reportes_clasificacion/mean_entropy_report_regiones.csv")
#%%
