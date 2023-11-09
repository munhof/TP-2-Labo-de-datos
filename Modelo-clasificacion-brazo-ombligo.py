#%%
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
    
Modelos de clasificacion remeras pantalones con knn, obtencion de zonas relevantes
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

#importo el data set
cod_prendas = pd.read_csv("cod-prendas.csv")

book_fotos = Limpieza_de_datos.book_fotos()
# grafico el promedio de las remeras

#%%
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

axs[0] = sns.scatterplot(data = data_test, x = "pixeles_brazos", y = "pixeles_ombligo", 
                         hue = "label", ax = axs[0] )
axs[0].set_title("distribucion datos original")

axs[1] = sns.scatterplot(data = data_test_modelo, x = "pixeles_brazos", y = "pixeles_ombligo",
                         hue = "label",  ax = axs[1] )
axs[1].set_title("distribucion datos modelo")
fig.suptitle("Modelo de clasficiacion de KNN por \n distancia arquetipos")

#borro las varibales
del fig,axs,data_test_modelo,data_test,knn_distancia,x_test,x_train

#%%
#ahora comprobemos como se comporta el modelo variando el hyper parametro n_neighbors

X = pd.concat([X_train,X_test])
y = pd.concat([y_train,y_test])

#armo una lista de parametros, en este caso distruidos con la secuencia de fibonacci
n = 145  # Reemplaza 10 con el valor deseado de 'n'
fibonacci = [0,1]
fibonacci = [fibonacci[i - 1] + fibonacci[i - 2] for i in range(2, n) if (fibonacci := fibonacci + [fibonacci[-1] + fibonacci[-2]])[-1] < n]

#defino hyperparametros
hyper_params = {'n_neighbors': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]}

#defino el modelo
knn_regiones = KNeighborsClassifier()
clf = GridSearchCV(knn_regiones, hyper_params,scoring="accuracy",cv = 5)
search = clf.fit(X[["pixeles_brazos","pixeles_ombligo"]], y)
# lo mejores parametros encontrados son:
print(f"mejor parametro {search.best_params_}")
mejor_param = search.best_params_["n_neighbors"]
mejor_score = search.best_score_
print(f"mejor score {search.best_score_}")

x_train = transfromadorBrazoOmbligo(X_train)
x_test = transfromadorBrazoOmbligo(X_test)

clf = KNeighborsClassifier(n_neighbors=5) 
clf.fit(x_train,y_train,)

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

primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
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
# %%
