# Construyo un algoritmo que se fije cuanto es la desviacion estandar dentro de 2 clases, agarre los 30 pixeles que
# tengan std mas grande (dado que son los que mas cambian de una prenda a la otra, tienen mas informacion) y mida
# la correlacion para cada par posible de pixeles (dado que no me sirve agarrar 3 pixeles juntos porque si uno esta
# claro los vecinos van a tender a ser claros tambien, no tengo que poder inferir la iluminacion del tercer pixel
# en base a los otros 2 porque sino dicho pixel es informacion redundante y sobra) para diferenciar si una imagen
# pertenece a la primera o segunda clase.

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










# Entreno y testeo un modelo de knn seleccionando los pixeles con el metodo visto anteriormente solo 
# con pantalones y remeras.

x_train = X_train[pixeles_elejidos]
x_test = X_test[pixeles_elejidos]

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

clf = KNeighborsClassifier(n_neighbors=17) 
clf.fit(x_train,y_train,)

fpr, tpr, _ = precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
plt.title("Precision-Recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
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










# Entreno y testeo un modelo de arbol seleccionando los pixeles con el metodo visto anteriormente con el dataset entero.

X_train,X_test,y_train,y_test = data_train_modelo_multiclase()

x_train = X_train[pixeles_elejidos]
x_test = X_test[pixeles_elejidos]

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

plt.show()
