# Construyo un algoritmo que se fije cuanto es la desviacion estandar dentro de 2 clases, agarre los 30 pixeles que 
# tengan std mas grande (dado que son los que mas cambian de una prenda a la otra, tienen mas informacion) y mida
# la correlacion para cada par posible de pixeles (dado que no me sirve agarrar 3 pixeles juntos porque si uno esta
# claro los vecinos van a tender a ser claros tambien, no tengo que poder inferir la iluminacion del tercer pixel
# en base a los otros 2 porque sino dicho pixel es informacion redundante y sobra) para diferenciar si una imagen 
# pertenece a la primera o segunda clase.
# Divido en train y test.

from sklearn.model_selection import train_test_split
X = dataset.iloc[:,1:785]
Y = dataset.iloc[:,  0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1) 
del X
del Y
del dataset

train = X_train.copy() # Defino una variable nueva para guardar el 'label' en la misma tabla.
train['label'] = Y_train

pantalones = train[train['label'] == 6] # Usare las clases pantalones y remeras como ejemplo. 
remeras = train[train['label'] == 0]
del train

consultaSQL = """
SELECT *
FROM pantalones
UNION
SELECT *
FROM remeras
"""

pantalones_y_remeras = sql^consultaSQL
del pantalones
del remeras
del consultaSQL

idex_de_pixeles_candidato = list (pantalones_y_remeras.std().nlargest(n=30).index) # Me da los 30 pixeles con mayor
# std entre las dos clases en forma de lista, los llamo candidato porque pasaron el filtro del std.

pixeles_candidato = pantalones_y_remeras[ idex_de_pixeles_candidato ] # Tengo que calcular correlacion, necesito
# la informacion de la iluminacion de los candidatos en cada imagen.
del pantalones_y_remeras
del idex_de_pixeles_candidato

from sklearn import linear_model
from sklearn.metrics import r2_score
# Dado el scatterplot de un pixel contra otro estos no van a estar correlacionados si dicho grafico esta homogeneamente
# cubierto de puntos, si estan concentrados en un cono entonces lo estan, la idea es hacer un ajuste lineal y calcular
# el error del ajuste, quiero que el error sea muy grande porque eso quiere decir que el ajuste no es significativo,
# entonces no se puede calcular la iluminacion de un pixel agarrando otro y haciendo una cuenta.


def correlacion_entre_pixeles (pixeles_candidato) :
    columnas = pixeles_candidato.columns 
    cant_candidatos = len( columnas )
    pixel1 = []
    pixel2 = []
    error = []  # error[k] es el error del ajuste lineal entre el pixel1[k] y pixel2[k].
    for i in range ( cant_candidatos - 1 ) :
        primer_pixel = columnas[i]
        Info_primer_pixel = pixeles_candidato[[primer_pixel]]
        for j in range ( i+1, cant_candidatos ) :
            segundo_pixel = columnas[j]
            info_segundo_pixel = pixeles_candidato[[segundo_pixel]]
            rls = linear_model.LinearRegression()
            rls.fit(Info_primer_pixel, info_segundo_pixel) # Hago el ajuste lineal, ahora me falta el error de dicho ajuste.

            def error_lineal_medio (ajuste_lineal, primer_pixel, segundo_pixel) :
                res = 0
                for i,j in zip(Info_primer_pixel[primer_pixel], info_segundo_pixel[segundo_pixel]) :
                    res = res + np.linalg.norm( ajuste_lineal(i) - j ) 
                    # Igual al error cuadratico medio pero sin la raiz ni el cuadrado, solo el modulo: Dado un punto
                    # calculo la distancia a la recta solo en el eje y, sumo al resultado y repito con todos los puntos.
                res = res/(Info_primer_pixel.shape[0])  # Solo me quedo con el promedio, sino queda un numero demasiado grande.
                return res

            def ajuste_lineal (x) :    # La recta que salio del ajuste lineal.
                a = rls.coef_[0][0]
                b = rls.intercept_[0]
                y = a*x + b
                return y
            
            pixel1.append(primer_pixel)
            pixel2.append(segundo_pixel)
            su_error = error_lineal_medio (ajuste_lineal, primer_pixel, segundo_pixel)
            error.append(su_error)
    tabla_de_correlacion = pd.DataFrame(np.column_stack([pixel1, pixel2, error]), columns=['Primer Pixel', 'Segundo Pixel', 'Su error'])
    # En cada fila de esta tabla hay dos pixeles y su correlacion, sin repetir.
    return tabla_de_correlacion

p=correlacion_entre_pixeles (pixeles_candidato) # Agarro la tabla y procedo a quedarme con los que tengan la correlacion mas grande.
k=(p['Su error']).max()
i=p[p['Su error']==k]

"""Hay un error en la ultima parte, por alguna razon cuando calculo el maximo de la tabla me tira una correlacion de 
como 10, siendo que si abro la tabla y pongo para que se ordene de menor a mayor veo claramente como la mas grande es 
en realidad de casi 100 (depende de como te divida entre train y test al principio).
Proba ordenando de menor a mayor y dividiendo el falso mas grande por el tercero mas grande (que si es el mas grande)
a ver que te da, xq se volvio loco, no se si el max funciona mal o es un bug visual y cuando me da el mas grande
se come un cero, no se, pero que la correlacion mas grande en los pixeles con mas std sea de 10 es absurdo."""