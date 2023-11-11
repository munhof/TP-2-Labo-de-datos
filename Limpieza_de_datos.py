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
    
Limpieza de los data-frame
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("./Dataset-original/fashion-mnist.csv")
#Vamos a crear los dataframes para cada uno de los modelos y tambien sus set de
#tablas para train-test-validacion
#si bien en al archivo exploracion de datos vimos que es posible eliminar 
#realmente no es un cantidad significativa, por lo que vamos a buscar metodos
#para encontrar cuales son las mas importante para nuestros objetivos
#en vez de ver cuales son irrelavantes, por eso se opta por no eliminar columnas

def data_validacion_modelo_pantalon_remera():
    """
    Genera los dataframe para validar el modelo de pantalones o remera
    No recibe parametros de entrada
    Devuelve los dataframe X_validacion, Df_resto y la serie
    Y-Validacion
    
    Returns
    -------
    data_validacion : (pd.dataframe,pd.series)
        DESCRIPTION 
        tupla pd.dataframe,pd.series que forman el conjunto de datos de 
        validacion para el modelo pantalon-remera
        .
    Df_rest : pd.dataframe
        DESCRIPTION
        resto de la data para entrenar el modelo de pantalon remera.

    """
    global dataset
    filtro_pantalon_remera = dataset["label"] <= 1 
    Df = dataset[filtro_pantalon_remera]
    res = Df[["label"]]
    #Separo del dataframe en validacion y el resto
    Df_rest, Df_validacion, = train_test_split(
            Df, test_size=0.1, random_state=3, stratify=res)
    X_validacion = Df_validacion.drop(columns = "label")
    y_validacion = Df_validacion[["label"]]
    data_validacion = (X_validacion,y_validacion)
    return data_validacion, Df_rest

def data_train_modelo_pantalon_remera():
    """
    Genera los dataframe para entrenar el modelo de pantalones o remera
    No recibe parametros de entrada
    Devuelve los dataframe X_train, X_test y la serie
    Y_train,Y_test

    Returns
    -------
    X_train : pd.dataframe
        DESCRIPTION.
        pd.dataframe con la informacion para entrenar el modelo
    X_test : TYPE
        DESCRIPTION. 
        pd.dataframe con la informacion para testear el modelo
    y_train : TYPE
        DESCRIPTION.
        pd.serie con la informacion para entrenar el modelo
    y_test : TYPE
        DESCRIPTION.
        pd.serie con la informacion para testear el modelo

    """
    _, Df_rest = data_validacion_modelo_pantalon_remera()
    #separo el resto en train y test
    res_rest = Df_rest[["label"]]
    Df_train,Df_test = train_test_split(
            Df_rest, test_size=0.3, random_state=3, stratify=res_rest)
    X_train = Df_train.drop(columns = "label")
    y_train = Df_train[["label"]]
    X_test = Df_test.drop(columns = "label")
    y_test = Df_test[["label"]]
    return X_train, X_test, y_train, y_test
    

def data_validacion_modelo_multiclase():
    """
    Genera los dataframe para validar el modelo multiclase
    No recibe parametros de entrada
    Devuelve los dataframe X_validacion, Df_resto y la serie
    Y-Validacion
    
    Returns
    -------
    data_validacion : (pd.dataframe,pd.series)
        DESCRIPTION 
        tupla pd.dataframe,pd.series que forman el conjunto de datos de 
        validacion para el modelo pantalon-remera
        .
    Df_rest : pd.dataframe
        DESCRIPTION
        resto de la data para entrenar el modelo de pantalon remera.

    """
    global dataset
    Df = dataset
    res = Df[["label"]]
    #Separo del dataframe en validacion y el resto
    Df_rest, Df_validacion, = train_test_split(
            Df, test_size=0.1, random_state=3, stratify=res)
    X_validacion = Df_validacion.drop(columns = "label")
    y_validacion = Df_validacion[["label"]]
    data_validacion = (X_validacion,y_validacion)
    return data_validacion, Df_rest

def data_train_modelo_multiclase():
    """
    Genera los dataframe para entrenar el modelo multiclase
    No recibe parametros de entrada
    Devuelve los dataframe X_train, X_test y la serie
    Y_train,Y_test

    Returns
    -------
    X_train : pd.dataframe
        DESCRIPTION.
        pd.dataframe con la informacion para entrenar el modelo
    X_test : TYPE
        DESCRIPTION. 
        pd.dataframe con la informacion para testear el modelo
    y_train : TYPE
        DESCRIPTION.
        pd.serie con la informacion para entrenar el modelo
    y_test : TYPE
        DESCRIPTION.
        pd.serie con la informacion para testear el modelo

    """
    _, Df_rest = data_validacion_modelo_multiclase()
    #separo el resto en train y test
    res_rest = Df_rest[["label"]]
    Df_train,Df_test = train_test_split(
            Df_rest, test_size=0.3, random_state=3, stratify=res_rest)
    X_train = Df_train.drop(columns = "label")
    y_train = Df_train[["label"]]
    X_test = Df_test.drop(columns = "label")
    y_test = Df_test[["label"]]
    
    return X_train, X_test, y_train, y_test


#tambien voy a generar las diferentes prendas arquetipicas

def prendas_arquetipicas():
    """
    funcion que devuelve un diccionarios con los arquetipos de cada prenda
    utilizando la mediana de cada pixel

    Returns
    -------
    arquetipos : dict[int]
        DESCRIPTION.
        Es un diccionario con 10 elementos donde cada uno tiene la imagen 
        arquetipica de cada prenda donde cada pixel tiene el valor de la mediana
        de los pixeles de todas las prendas de una misma clase

    """
    global dataset
    arquetipos = {}
    for i in range(10):
        arquetipos[i] = dataset[dataset["label"] == i].drop(columns = "label").mean()
    return arquetipos

def book_fotos():
    global dataset
    book_fotos = {}
    for cod in range(10):
        book_fotos[cod] = dataset[dataset["label"] == cod]
    return book_fotos







