# [Detector-de-canchas](detector.py)

Este detector de canchas consiste en dos fases: 
1. [Detectar posibles canchas calculando los anchos del camino.](#detector-por-anchos)
2. [Pasar las posibles canchas por una red neuronal.](#red-neuronal-alexnet)

El algoritmo principal se encuentra en [detector.py](detector.py). Para usarlo se necesita la imagen del terreno en .tif y un shape file que delimite los caminos.

---

## [Detector por anchos](DCPA.py)
Resumen: Se recorren los caminos dados por el shape file y entre dos vértices se crea una línea perpendicular, con la cual se calcula el ancho del camino.

### Funciones usadas:
- [Detectar las canchas](#detCanchasAncho)
- [Juntas los centros cercanos](#postJuntarPuntos)

### detCanchasAncho
- Parámetros: 
    - **caminos**: ShapeFile que describe los caminos del predio (en general con TIPOUSO = FCAM).
    - **lengthComp**: Valor entero que expresa la cantidad de comparaciones que se harán, si se cumple un aumento de ancho en estas comparaciones, entonces se considerará como cancha. (valor default: **lengthComp = 3**)
    - **sensProp**: Float que indica la proporción que debe crecer el camino para que sea considerado cancha. (valor default: **sensProp = 1.1**)


- Retorna: Lista de Points de Shapely que indican el centro de las canchas detectadas.

La función calcula los anchos creando líneas entre puntos y detecta si el camino aumenta en su ancho en un sensProp\*100, si mantiene ese aumento por lengthComp comparaciones seguidas, se marca el lugar como cancha.

- Como afecta alterar el lengthComp:
    - Al usar valores bajos, por ejemplo 1 o 2, el detector se vuelve muy sensible y detecta como canchas cambios espontáneos de anchos, como en bifurcaciones o curvas.
    - Al usar valores muy altos, por ejemplo mayores a 10, puede ocurrir que no se detecten canchas que la describen una cantidad de puntos menores a lengthComp.
- Como afecta alterar el sensProp:
    - En valores bajos, como 1.1 o menos, es posible que se detecten muchas canchas falsas, especialmente en las curvas, donde crece levemente el ancho.
    - Con valores muy altos no se detectan canchas muy delgadas o que crecen muy poco en comparación con el camino con el cual se conecta.
- Valores default:
    - Experimentalmente, con los valores de lengthComp = 3 y sensProp = 1.1, se detectan alrededor de un 98% de las canchas, pero a su vez, se obtienen muchos dalsos positivos.
    
### postJuntarPuntos
- Parámetros:
    - **caminos**: ShapeFile que describe los caminos del predio (en general con TIPOUSO = FCAM).
    - **centros**: Lista de Points de Shapely, siendo el centro de las canchas detectadas.
    - **separateTol**: Valor entero que describe la distancia máxima a la cual se decide juntar los puntos.
    
- Retorna: Lista de Points de shapely de canchas detectadas, con centros repetidos eliminados.

La función "mezcla" los puntos que se encuentran a separateTol de distancia siempre y cuando la línea que conecta estos puntos se encuentra dentro del mismo camino, de esta forma se consigue eliminar múltiples detecciones para la misma cancha. Se recomienda usarla con el resultado de **detCanchasAncho()**, la cual es susceptible a detectar múltiples veces la misma cancha, sobre todo con baja sensibilidad.

---

## [Red neuronal AlexNet](finetune_alexnet)
**Código usado de AlexNet: [finetune_alexnet](https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d)**

### [cortar_imagen](funciones.py)
El proceso de las canchas consiste en tomar los puntos de las canchas y crear un cuadrado, cuyo centro es el punto marcado, para luego ubicarlo en la imagen del predio y cortar la imgen.

- Parámtetros: 
    - **file_tif**: Dirección en donde se encuentra el .tif del predio en el computador.
    - **region_name**: Nombre del predio (número), usado para nombrar las imagenes.
    - **save_file**: Dirección en donde se van a guardar las imágenes.
    - **inicio**: Índice para empezar a enumerar las canchas, sirve para nombrarlas.
    - **tipo**: Tipo de uso de la imgen, puede ser 'train', 'valid', 'test', 'predict'. (default: tipo = 'predict')
    - **canchas**: Lista de Points de Shapely, siendo el centro de las canchas detectadas.
    - **img_normal**: *True* si se quiere recortar la imagen de la cancha, *False* solo tiene sentido si ya se tiene la imagen y se quiere utilizar img_flip. (default: img_normal = True)
    - **img_flip**: *True* si se quiere recortar la imagen reflejada en el eje horizontal y vertical, utilizado para obtener imágenes para entrenar la red. (default: img_flip = False)
    
    
- Retorna: Un diccionario de tipo \[Nombre_de_imagen.jpg\] = id correspondiente a la cancha en **canchas**.

### [usar_red](funciones.py)
Usa un [modelo previamente entrenado](finetune_alexnet/tmp) para hacer las predicciones sobre las imágenes recortadas. Las imágenes a detectar deben estar en una carpeta llamada *Predict* en esta misma carpeta.

- Parámetros:
    - **txt**: Nombre del archivo en formato .txt para guardar las predicciones.
    - **num_modelo**: Número del modelo entrenado, se utiliza para saber su nombre. El número está relacionado con la cantidad de *epochs* con el que fue entrenado el modelo. Los modelos se encuentran en [tmp](finetune_alexnet/tmp)

- Retorna: Lista con el nombre de las imágenes que **si** son canchas.
