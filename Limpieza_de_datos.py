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
    
Limpieza de los data-frame
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importamos el modulo pyplot, y lo llamamos plt
import seaborn as sns

dataset = pd.read_csv("./Dataset-original/fashion-mnist.csv")
