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
plt.savefig('.\modelo-pantalon-remera\Clasificacion-arquetipos-arbitraria.png')

#borro las varibales
del fig,axs,data_test_modelo,data_test,knn_distancia,x_test,x_train,
del pantalon_arquetipo, remera_arquetipo
del distancia_pantalon, distancia_remera
#ahora comprobemos como se comporta el modelo variando el hyper parametro n_neighbors

x_train = X_train[["distancia remera","distancia pantalon"]]
x_test = X_test[["distancia remera","distancia pantalon"]]
x_test_distancia = x_test.copy()


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

knn_distancia = KNeighborsClassifier(mejor_param) 
knn_distancia.fit(x_train,y_train,)
data_test = x_test.copy()
data_test["label"] = y_test.copy()

data_test_modelo = x_test.copy()
data_test_modelo["label"] = knn_distancia.predict(x_test)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axs[0] = sns.scatterplot(data = data_test, x = "distancia remera", y = "distancia pantalon", 
                         hue = "label", ax = axs[0] )
axs[0].set_title("distribucion datos original")

axs[1] = sns.scatterplot(data = data_test_modelo, x = "distancia remera", y = "distancia pantalon",
                         hue = "label",  ax = axs[1] )
axs[1].set_title("distribucion datos modelo")
fig.suptitle("Modelo de clasficiacion de KNN por \n distancia arquetipos")
plt.savefig('.\modelo-pantalon-remera\Clasificacion-arquetipos-mejor.png')

y_pred_prob = knn_distancia.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
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


primos = [2, 3, 5,  13, 17, 19, 23, 29, 41,53 ,67, 71, 73, 79,]
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
ax1.set_xlabel("K-vecinos")
ax1.set_ylabel("Performance")
ax1.set_title("Puntajes de CrossValidation del modelo\n para diferentes hiperparámetros")
ax1.grid()

# Gráfico de curvas ROC
for i,param  in enumerate(primos_dict):
    fpr, tpr = roc_curves[i]
    ax2.plot(fpr, tpr, label=f" k = {i}, AUC={roc_aucs[i]:.3f}")

ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.set_title('Curvas ROC')
ax2.legend(title = "Cantitad de vecinos y AUC")
ax2.grid()
plt.tight_layout()
plt.savefig('.\modelo-pantalon-remera\Clasificacion-arquetipos-evaluacion.png')
plt.show()
#%%
#aca inicia el modelo de correlacion
from Limpieza_de_datos import data_train_modelo_pantalon_remera, data_train_modelo_multiclase
from sklearn import linear_model
from sklearn.metrics import r2_score, classification_report, roc_curve, roc_auc_score,precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
#%%
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
x_test_pix = x_test.copy()
#armo el modelo, arbitrariamente elijo 5
knn_correlacion = KNeighborsClassifier(n_neighbors=5) 

#entreno el modelo
knn_correlacion.fit(x_train,y_train)


#reviso el score con los dato de test
print(knn_correlacion.score(x_test, y_test))


#defino hyperparametros
hyper_params = {'n_neighbors': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]}

#defino el modelo
knn_correlacion = KNeighborsClassifier()
clf = GridSearchCV(knn_correlacion, hyper_params,cv = 5)
search = clf.fit(x_train, y_train)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param_correlacion = search.best_params_["n_neighbors"]
mejor_score_correlacion = search.best_score_
print(f"mejor score {search.best_score_}")

knn_correlacion = KNeighborsClassifier(mejor_param_correlacion) 
knn_correlacion.fit(x_train,y_train,)
y_pred_prob = knn_correlacion.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.title("Precision-Recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")

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

primos = [2, 3, 5,  13, 17, 19, 23, 29, 41,53 ,67, 71, 73, 79]
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
ax1.axhline(mejor_score_correlacion,linestyle = ":", color = "green" ,label = "Mejor score medio obtenido en CV")
ax1.axvline(mejor_param_correlacion,linestyle = ":", color = "green" ,label = "Mejor parametro encontrado en CV")
ax1.legend()
ax1.set_xlabel("K-vecinos")
ax1.set_ylabel("Performance")
ax1.set_title("Puntajes de CrossValidation del modelo\n para diferentes hiperparámetros")
ax1.grid()

# Gráfico de curvas ROC
for i,param  in enumerate(primos_dict):
    fpr, tpr = roc_curves[i]
    ax2.plot(fpr, tpr, label=f" k = {i}, AUC={roc_aucs[i]:.3f}")

ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.set_title('Curvas ROC')
ax2.legend(title = "Cantitad de vecinos y AUC")
ax2.grid()
plt.tight_layout()
plt.savefig('.\modelo-pantalon-remera\Clasificacion-pixeles-evaluacion.png')
plt.show()
#%%
#aca incia el modelo de regiones
book_fotos = Limpieza_de_datos.book_fotos()
dataset_shirt = book_fotos[0]
remera_arquetipo = Limpieza_de_datos.prendas_arquetipicas()[0].to_numpy()

dataset_trouser = book_fotos[1]
pantalon_arquetipo = Limpieza_de_datos.prendas_arquetipicas()[1].to_numpy()

# grafico la diferencia absoluta entre el prom. de las dos imagenes
# esto me muestra las zonas de que mas diferencian una remeras de un pantalón
distancia_arquetipos = np.absolute(remera_arquetipo-pantalon_arquetipo)



# Creo un dataframe con los datos de remeras y pantalones
data_remera =Limpieza_de_datos.book_fotos()[0]
data_pantalon =Limpieza_de_datos.book_fotos()[1]
data_remera_pantalon = pd.concat([data_remera,data_pantalon])

#%%
#Verifico que esten balanceados
data_remera.info()
data_pantalon.info()
data_remera_pantalon.info()

#%%
# Selecciono los pixeles de interes
print(distancia_arquetipos.max())
print(np.where(distancia_arquetipos >= 140))
#%%
# pixeles de interes
#  pixeles_brazos : [203,231,232,246,260]
#  pixeles_ombligo : [575,603,631,659,687,743]

#%%
# me quedo con los pixeles de interes y los sumo para asignarlos a los atributos brazo y ombligo
def transfromadorBrazoOmbligo(data):
    data['pixeles_brazos'] = data[['pixel203','pixel231','pixel232','pixel246','pixel260']].sum(axis=1)/255
    data['pixeles_ombligo'] = data[['pixel575','pixel603','pixel631','pixel659','pixel687','pixel743']].sum(axis=1)/255
    data = pd.DataFrame(data[['pixeles_brazos','pixeles_ombligo']])
    return data

#defino los dataset de train y test
X_train, X_test, y_train, y_test = Limpieza_de_datos.data_train_modelo_pantalon_remera()

x_train = transfromadorBrazoOmbligo(X_train)
x_test = transfromadorBrazoOmbligo(X_test)

#armo el modelo, arbitrariamente elijo 5
knn_regiones = KNeighborsClassifier(n_neighbors=5) 

#entreno el modelo
knn_regiones.fit(x_train,y_train)

data_test = x_test.copy()
data_test["label"] = y_test.copy()

data_test_modelo = x_test.copy()
data_test_modelo["label"] = knn_regiones.predict(x_test)

#reviso el score con los dato de test
print(knn_regiones.score(x_test, y_test))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axs[0] = sns.scatterplot(data = data_test, x = "pixeles_brazos", y = "pixeles_ombligo", 
                         hue = "label", ax = axs[0] )
axs[0].set_title("distribucion datos original")

axs[1] = sns.scatterplot(data = data_test_modelo, x = "pixeles_brazos", y = "pixeles_ombligo",
                         hue = "label",  ax = axs[1] )
axs[1].set_title("distribucion datos modelo")
fig.suptitle("Modelo de clasficiacion de KNN por \n distancia arquetipos")
plt.savefig(".\modelo-pantalon-remera\clasificacion-brazo-ombligo-arbitraria.png")
#borro las varibales
del fig,axs,data_test_modelo,data_test,x_test,x_train

#%%
#ahora comprobemos como se comporta el modelo variando el hyper parametro n_neighbors

X = pd.concat([X_train,X_test])
y = pd.concat([y_train,y_test])


#defino hyperparametros
hyper_params = {'n_neighbors': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]}

#defino el modelo
knn_regiones = KNeighborsClassifier()
clf = GridSearchCV(knn_regiones, hyper_params,scoring="accuracy",cv = 5)
search = clf.fit(X[["pixeles_brazos","pixeles_ombligo"]], y)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param_regiones = search.best_params_["n_neighbors"]
mejor_score_regiones = search.best_score_
print(f"mejor score {search.best_score_}")

x_train = transfromadorBrazoOmbligo(X_train)
x_test = transfromadorBrazoOmbligo(X_test)
x_test_region = x_test.copy()
knn_regiones = KNeighborsClassifier(mejor_param_regiones) 

#entreno el modelo
knn_regiones.fit(x_train,y_train)

data_test = x_test.copy()
data_test["label"] = y_test.copy()

data_test_modelo = x_test.copy()
data_test_modelo["label"] = knn_regiones.predict(x_test)

#reviso el score con los dato de test
print(knn_regiones.score(x_test, y_test))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axs[0] = sns.scatterplot(data = data_test, x = "pixeles_brazos", y = "pixeles_ombligo", 
                         hue = "label", ax = axs[0] )
axs[0].set_title("distribucion datos original")

axs[1] = sns.scatterplot(data = data_test_modelo, x = "pixeles_brazos", y = "pixeles_ombligo",
                         hue = "label",  ax = axs[1] )
axs[1].set_title("distribucion datos modelo")
fig.suptitle("Modelo de clasficiacion de KNN por \n distancia arquetipos")
plt.savefig(".\modelo-pantalon-remera\clasificacion-brazo-ombligo-mejor.png")

fpr, tpr, _ = precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
plt.plot(fpr,tpr)
#sin embargo verifico la performance de cada uno para ver como se comporta

#%%
def evaluate_model(model, x_train,x_test, y_train,y_test, hyperparameters, cv):
    # Listas para almacenar los resultados
    score_best = []
    score_worse = []
    score_mean = []
    roc_curves = []
    roc_aucs = []


    for param in hyperparameters:
        clf = model(**param)
        # Calcula los puntajes de validación cruzada
        score_cv = cross_val_score(clf, x_train, y_train, cv=cv)
        score_best.append(max(score_cv))
        score_worse.append(min(score_cv))
        score_mean.append(score_cv.mean())
        
        # Listas para almacenar las predicciones y etiquetas reales
        clf.fit(x_train, y_train)
        
        y_pred_prob = clf.predict_proba(x_test)[:, 1]
        
            
        # Después de completar todos los folds
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # Calcula la curva ROC
        roc_curves.append((fpr, tpr))
        
        # Calcula el área bajo la curva ROC (AUC)
        auc = roc_auc_score(y_test, y_pred_prob)
        roc_aucs.append(auc)
    
    
    return score_best,score_worse,score_mean,roc_curves,roc_aucs

primos = [2, 3, 5,  13, 17, 19, 23, 29, 41,53 ,67, 71, 73, 79,]
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
ax1.axhline(mejor_score_regiones,linestyle = ":", color = "green" ,label = "Mejor score medio obtenido en CV")
ax1.axvline(mejor_param_regiones,linestyle = ":", color = "green" ,label = "Mejor parametro encontrado en CV")
ax1.legend()
ax1.set_xlabel("K-vecinos")
ax1.set_ylabel("Performance")
ax1.set_title("Puntajes de CrossValidation del modelo\n para diferentes hiperparámetros")
ax1.grid()

# Gráfico de curvas ROC
for i,param  in enumerate(primos_dict):
    fpr, tpr = roc_curves[i]
    ax2.plot(fpr, tpr, label=f" k = {i}, AUC={roc_aucs[i]:.3f}")

ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('Tasa de Falsos Positivos')
ax2.set_ylabel('Tasa de Verdaderos Positivos')
ax2.set_title('Curvas ROC')
ax2.legend(title = "Cantitad de vecinos y AUC")
ax2.grid()
plt.tight_layout()
plt.savefig('.\modelo-pantalon-remera\Clasificacion-region-evaluacion.png')
plt.show()
# %%
#en esta seccion comparamos los tres modelos con los mejores parametros obtenidos

y_pred_prob_distancia = knn_distancia.predict_proba(x_test_distancia)
y_pred_prob_correlacion = knn_correlacion.predict_proba(x_test_pix)
y_pred_prob_regiones = knn_regiones.predict_proba(x_test_region)

lista_predicciones = [y_pred_prob_distancia[:,1],
                      y_pred_prob_correlacion[:,1],
                      y_pred_prob_regiones[:,1]]
lista_nombres = ["Distancias","Correlaciones","Regiones"]

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test, lista_predicciones[i])
        
        # Calcula la curva ROC
        
        # Calcula el área bajo la curva ROC (AUC)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f" modelo = {lista_nombres[i]}, AUC={roc_aucs[i]:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC')
    plt.legend(title = "Cantitad de vecinos y AUC")
    plt.grid()
plt.savefig(".\modelo-pantalon-remera\curvas-roc-compartivas.png")

from Limpieza_de_datos import data_validacion_modelo_pantalon_remera

(x_validacion, y_validacion),_ = data_validacion_modelo_pantalon_remera()
x_validacion = transfromadorBrazoOmbligo(x_validacion)
y_pred_prob_validacion =  knn_regiones.predict_proba(x_validacion)[:,1]
fpr, tpr, _ = roc_curve(y_validacion, y_pred_prob_validacion)
auc = roc_auc_score(y_validacion, y_pred_prob_validacion)
plt.plot(fpr, tpr, label=f"AUC={roc_aucs[i]:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC')
plt.legend(title = "AUC")
plt.grid()
plt.savefig(".\modelo-pantalon-remera\curvas-roc-validacion.png")