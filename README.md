# Evaluación de rendimiento de arquitecturas paralelas y de propósito específico para el aprendizaje por refuerzo en juegos
## Trabajo de Fin de Grado en Ingeniería Informática
### Javier Guzmán Muñoz
### Doble Grado en Ingeniería Informática y Matemáticas 
### Universidad Complutense de Madrid

#### Memoria del proyecto [aquí](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/memoria.pdf)

Las aplicaciones de aprendizaje por refuerzo se usan en la actualidad para resolver problemas de todo tipo en campos muy diversos. Sin embargo, una de las principales desventajas que presentan es el elevado coste computacional del entrenamiento de los modelos necesarios. Con este trabajo de fin de grado se pretende mejorar este proceso mediante la paralelización de los algoritmos empleados y el uso de distintas arquitecturas hardware que variarán los tiempos empleados. Los modelos entrenados pueden aplicarse para obtener la mejor secuencia de acciones que podemos realizar sobre un entorno y que mejore la recompensa obtenida. Este proceso, que se denomina inferencia, es ya de por sí bastante eficiente en tiempo, pero la existencia de procesadores de propósito específico para realizar esta tarea hace también conveniente evaluar su rendimiento en estos soportes y compararlos con otras unidades de procesamiento más generales. Tras definir en el escenario en el que nos vamos a mover y los recursos necesarios para ello, se definen una serie de experimentos de los procesos de entrenamiento e inferencia que nos permitirán evaluar el rendimiento en términos del tiempo empleado, de la utilización de los recursos disponibles y del consumo de energía de distintas arquitecturas hardware, viendo cuál es más conveniente usar en cada caso.

El presente repositorio contiene el código necesario para llevar a cabo diferentes experimentos de entrenamiento e inferencia con una configuración especificada. Los entrenamientos se realizan dentro del framework de Ray, concretamente de su biblioteca de aprendizaje por refuerzo [RLlib](https://docs.ray.io/en/master/rllib.html) y usando el algoritmo PPO, que permite una paralelización de sus etapas para mejorar el rendimiento. Los procesos de inferencia, esto es, la aplicación de modelos previamente entrenados para obtener la mejor secuencia de acciones que maximicen la recompensa final, pueden llevarse a cabo usando la funcionalidad de RLlib. Adaptamos los scripts propios de la biblioteca para nuestro propósito, añadiendo funcionalidad que nos reporte información acerca del tiempo empleado. Además, ofrecemos los scripts necesario para llevar a cabo la transformación de un modelo entrenado con RLlib hata uno de Tensorflow Lite compilado para poder ser ejecutado en el acalerador Google Coral que incluye una TPU, un procesador de propósito específico para llevar a cabo inferencias de modelos de aprendizaje profundo. 

La memoria del proyecto contiene información detallada acerca de todo el proceso de experimentación que se lleva a cabo y de los resultados obtenidos. Además, se incluye una guía de uso de los principales scripts que componen este repositorio. Hay que tener en cuenta que los experimentos se realizan en un servidor que, entre otros recursos, cuenta con dos GPUs y los scripts están adaptados para este propósito.

# Uso de los principales scripts

El código ha sido probado y ejecutado con éxito en un entorno \textit{Python 3.7.3} con las siguientes librerías instaladas:

  - `Ray 1.1.0`
  - `Tensorflow 2.4.1`
  - `Gym 0.18.0`
  
Además, para obtener métricas relativas al uso de las GPUs ha sido necesario instalar la biblioteca `gputil` y para obtener la representación gráfica y en ficheros de los datos se emplean las bibliotecas `matplotlib` y `pandas`.
## Script de entrenamiento 
Proporcionamos un *script* [`train_ppo.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/train_ppo.py) con el que se pueden realizar varios experimentos de entrenamiento para los seis modelos con los que trabajamos. El *script* contiene, aparte de la función `main` tres funciones auxiliares:

  - `gpu_options(gpu_opt)`: establece las GPUs que se mostrarán visibles al proceso y por tanto podrán ser utilizadas para el entrenamiento. Recibe en `gpu_opt` un *string* indicando la configuración deseada: `gpu0` para usar únicamente la GPU con identificador 0 (RTX en el servdior *volta1*, `gpu1` para usar únicamente la GPU con identificador 1 (Tesla-v100 en *volta1*), `none` para no usar ninguna de ellas y `both` para usar las dos. Para ello, se establece el valor de la variable de entorno del sistema `CUDA_VISIBLE_DEVICES` con los identificadores de las GPUs que queremos que sean visibles en cada caso.
    
  - `get_config(model)`: Devuelve un diccionario con la configuración de un agente para cada uno de los seis modelos propuestos en la tabla \ref{tab:modelos}, que se especifican con un entero con valores entre 1 y 6 mediante el parámetro `model`.
  - `full_train(checkpoint_root, agent, n_iter, save_file, n_ini = 0, header = True, restore = False, restore_dir = None)`: ejecuta una serie de iteraciones de entrenamiento sobre un agente dado y devuelve una estructura con sus resultados, además de guardar esta información en unos ficheros `.csv` y `.json`. Recibe como argumentos:
    * `checkpoint_root`: *string* con la ruta del directorio en el que queremos que se vayan guardando los *checkpoints* para cada paso de entrenamiento realizado.
    * `agent`: agente de *RLlib* sobre el que ejecutar las iteraciones de entrenamiento.
    * `n_iter`: entero indicando el número de iteraciones de entrenamiento del algoritmo concreto del agente (en nuestro caso PPO) a ejecutar.
    * `save_file`: ruta al archivo en el que queremos que se almacenen la información del entrenamiento. Mediante un *string* indicamos la ruta a un archivo sin extensión, así se crearán dos archivos en esa ruta con extensiones `.json` y `.csv`.
    * `n_ini`: entero indicando el número de la última iteración realizada, su valor por defecto es 0, indicando que aun no hemos comenzado a entrenar ese modelo.
    * `header`: booleano indicando si hay que añadir la línea de cabecera con los nombres de las columnas al fichero `.csv` con los datos del entrenamiento. Su valor por defecto es `True` indicando que si es la primera vez que estamos entrenando el modelo sí hay que añadir esta línea.
    * `restore`: booleano indicando si debemos establecer o no el estado del agente desde un *checkpoint*, cuya ruta indicamos en `restore_dir`. Su valor por defecto es `False`.
    * `restore_dir`: ruta del *checkpoint* desde el que queremos restaurar el estado del agente, si hemos indicado `restore=True`.
  
La función devuelve una lista con un diccionario por cada iteración de entrenamiento, en el que se incluyen el número de iteración, las recompensas mínima, media y máxima de los episodios, la longitud media de los episodios, el tiempo de la fase de aprendizaje en ms Y el tiempo total en segundos de esa iteración. Estos mismos datos se guardan en los ficheros `.json` y `.csv` antes mencionados.

Así, para ejecutar uno de los experimentos de entrenamiento ejecutamos el *script* indicando pudiendo indicarle el valor de varios argumentos:
  - `-m, --model`: entero (1-6) indicando el identificador del modelo a entrenar.
  - `-g, --gpu`: string con los valores `gpu0, gpu1, none, both` indicando la configuración de GPUs con las que realizar el entrenamiento.
  - `-d, --driver-gpus`: número de GPUs que se asignarán al driver (`config[num_gpus]`), puede ser un número decimal. El resto se repartirán a partes iguales entre los *workers*.
  - `-w, --workers`: número de *workers* que se crearán en el algortimo para recoger experiencias del entorno.
  - `-s, --save-name`: ruta del fichero, sin extensión, en el que se guardarán los datos de entrenamiento en formatos `.json` y `.csv`.
  - `-i, --iters`: número de iteraciones de entrenamiento a ejecutar.
  - `-c, --cpus`: número de CPUs que indicamos a *Ray* en su inicialización. Su valor por defecto es `None`, que indica que *Ray* usará todas las que encuentre disponibles.
  - `-a, --set-affinity`: conjunto con los identificadores de las CPUs a las que queremos restringir la ejecución con `sched_setaffinity`. Su valor por defecto es el conjunto vacío (`{}`), que indica que no forzamos a que el programa se ejecute en unas CPUs concretas.
  - `-r, --restore-dir`: dirección del *chekpoint* desde el que queremos resturar el estado del agente. Su valor por defecto es `None` que indica que no queremos restaurar desde ningún *checkpoint*.
    
Además de realizar las iteraciones de entrenamiento indicadas, la ejecución de este *script* mueve los ficheros con las métricas que reporta Ray (y que por defecto se guardan en un directorio dentro de `~/ray_results` cuyo nombre viene dado por el *timestamp* del momento en que se inicia la ejecución) y los almacena en un directorio dentro de la carpeta `ray_results` del proyecto y con el nombre indicado por `save_name`. Además, también copia el fichero `params.pkl` de este directorio en el que se guardan los *checkpoints*, pues luego será necesario que este ahí para la ejecución de inferencias.

Por ejemplo, podemos ejecutar 1000 iteraciones de enyrenamiento para el modelo 3 usando sólo la GPU 0 del sistema, con 0.001 GPUs para el *driver* y 4 *workers* que se reparten el resto de la GPU con la siguiente instrucción:

```
$ python training_scripts/train_ppo.py --model=3 --gpu=gpu0 --driver-gpus=0.001 --workers=4 --save-name=model3_4_workers_gpu0 --iters=1000
```
Esto generará un directorio para cada *checkpoint* en `checkpoints/ppo/model3_4_workers_gpu0` y unos ficheros `training_results/ppo/model3_4_workers_gpu0.csv` y el mismo pero con extensión `.json` con algunos datos del entrenamiento. Además, tendremos en `ray_results/model3_4_workers_gpu0` los ficheros con las métricas que genera *Ray*.


## Script de inferencia en RLlib
El *script* [`rollout_with_time.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/rollout_with_time.py) será el que utilicemos para realizar los experimentos de inferencia en \textit{RLlib}. Este \textit{script} es una modificación del que proporciona ya [*RLlib*](https://github.com/ray-project/ray/blob/master/rllib/rollout.py), al que se le añade el código necesario para medir y guardar datos sobre el tiempo que se toma en cada inferencia y para la gestión de los recursos disponibles. Así, podemos especificar una serie de parámetros cuando ejecutemos este *script*, algunos de los cuales proviene del *script* original de *RLlib*:

- `checkpoint`: primer argumento, con él indicamos la ruta al checkpoint desde el que queremos restablecer el estado del agente para las inferencias.
- `--run`: algoritmo con el que hemos entrenado al agente. En nuestro caso siempre tomará el valor `PPO`.
- `--env`: entorno *Gym* sobre el que ejecutar las inferencias. En nuestro caso tomará el valor `Pong-v0`.
- `--time-output`: ruta a un fichero `.csv` en el que se guardarán los datos de tiempo de las inferencias.
- `--no-render`: es necesario añadir este argumento si no queremos que se muestre por pantalla las interacciones con el entorno. Nosotros siempre lo añadiremos.
- `--gpu`: configuración de GPUs con las que realizar la inferencia. Puede tomar los valores `gpu0`, `gpu1`, `none` y `both`.
- `--video-dir`: directorio en el que guardaremos videos de las interacciones. No lo utilizamos en este trabajo.
- `--seteps`: número de pasos de inferencia a ejecutar. Si especificamos un número de episodios (con `--episodes`) el valor que le hayamos dado al número de pasos quedará sin efecto.
- `--episodes`: número de episodios completos a ejecutar.
- `--config`: diccionario con la configuración del agente, que sobreescribe a la cargada del fichero `params.pkl` del directorio del *checkpoint*.
- `--save-info`: guarda información sobre las observaciones y las acciones de cada paso de inferencia. No lo utilizaremos.
- `--use-shelve`: guarda la información sobre las observaciones y las acciones de cada paso de inferencia con formato *shelf*.
- `--set-affinity`: Conjunto (*set*) con los identificadores de las CPUs a las que queremos restringir la ejecución.
- `num-cpus-ray` número de CPUs que indicamos a *Ray* en su incicialización. Si su valor es 0 (lo es por defecto), le estamos indicando a *Ray* que puede usar todas las que encuentre disponibles.

La configuración de recursos específica (número de *workers*, GPUs para el *driver*...) podemos especificarla en le parámetro `--config`. Un ejemplo de ejecución de inferencia sin GPUs y sin crear *workers* sería:
```
$ python rollout_with_time.py checkpoints/ppo/model1_gpu/checkpoint_11000/checkpoint-11000 --run=PPO --env=Pong-v0 --time-output=rollout_results/volta1/model1_no_gpus_0_workers.csv --no-render --gpu=none --episodes=10 --config='{"num_workers":0, "num_gpus_per_worker":0, "num_gpus":0}
```
## Scripts de exportación y cuantización de modelos para la TPU
Detallaremos ahora el contenido y manera de uso de los cuatro *scripts* que llevan a cabo el proceso completo de creación de modelos de *Tensorflow Lite* cuantizados que pueden ser ejecutados en la TPU.
*Script de exportación de modelos*
El *script* [`model_saver.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/model_saver.py) parte de un modelo entrenado en *RLlib* y exporta la red neuronal con la que se modela la política y su valor en formato `.h5`. Para ello, la ejecución del *script* requiere dos parámetros en su llamada:

  - Dirección a un *checkpoint* desde el que restableceremos el estado del agente a exportar.
  - Ruta donde queremos guardar el modelo en formato `.h5`. Se indicará la ruta al fichero y su nombre sin extensión.

El *script* creará un agente PPO restaurando el estado del *heckpoint* pasado como primer argumento y guardará el modelo de *keras* que contiene la red neuronal de la política y su valor en un fichero con extensión `.h5` en la dirección especificada como segundo argumento. Por ejemplo, podemos obtener un fichero `.h5 del modelo 1 ejecutando:
```
$ python model_saver.py checkpoints/ppo/model1_gpu/checkpoint_1000/checkpoint-1000 exported_models/model1
```
### Script de creación de modelos de Tensorflow Lite
El *script* [`tflite_converter.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/tflite_converter.py) crea y guarda un modelo de *Tensorflow Lite* a partir de un modelo de *keras* previamente exportado en formato `.h5`. En su ejecución debemos indicarle el valor de dos arumentos:
  - Dirección del rachivo con extensión `.h5` donde se encuentra el modelo de *keras* exportado.
  - Dirección del fichero `.tflite` con extensión donde queremos guardar el modelo resultante.


El *script* creará un objeto `TFLiteConverter` que llevará a cabo la conversión a partir del modelo de *keras* previamente cargado. Por ejemplo, para crear un modelo de *Tensorflow Lite* del modelo 1 podemos ejecutar:
```
$ python tflite_converter.py exported_models/model1.h5 exported_models/model1.tflite
```
### Script de creación de datasets para la cuantización
El *script* [`dataset_creator.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/dataset_creator.py) crea y guarda conjuntos de imágenes del entorno con el que interaccionan los modelos y que toman como entradas y que son necesarias para que durante el proceso de cuantización se puedan estimar los rangos que toman los tensores de entrada y de salida del modelo (pues sus valores son variables) y el modelo cuantizado pierda la menor precisión posible respecto al original. Debemos especificar el valor de dos argumentos en la ejecución del *script*:
  - Dimensión de las imágenes que guardaremos en el \textit{dataset}.
  - Ruta en la que se guardará el \textit{dataset} que se cree, sin extensión.


Una vez ejecutemos el *script*, se creará un entorno como con el que interaccionan los agentes y se tomarán 500 imágenes obtenidas como observaciones tras ejecutar una serie de acciones aleatorias sobre este entorno. Estas imágenes se guardarán en un fichero con extensión `.npy` (pues son en realidad `arrays` de *Numpy*) en la ruta indicada como segundo argumento. Por ejemplo, podemos crear un *dataset* con imágenes de dimensión `168 x 168 x 4`, que podrían ser usado para la cuantización del modelo 4, ejecutando:
```
$ python dataset_creator.py 168 datasets/dataset_model4
```

### Script de cuantización de modelos de Tensorflow Lite
El *script* [`quantizer.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/quantizer.py) lleva a cabo la creación de un modelo de *Tensorflow Lite* cuantizado, con todos su parámetros como enteros de 8 bits, a partir de un modelo de *keras* exportado en un fichero `.h5`. Para ello requerirá tres argumentos cuando lo ejecutemos:

  - Dirección a un dataset, con extensión `.npy` que contenga al menos 500 imágenes que podrían ser entrada del modelo que queremos convertir.
  - Dirección del modelo de \textit{keras} con extensión `.h5` que queremos convertir a *Tensorflow Lite* y cuantizar.
  - Dirección del fichero con extensión `.tflite` donde queremos guardar el modelo convertido a *Tensorflow Lite* y cuantizado.


El *script* contiene la función `representative_data_gen()` que toma 100 imágenes del *dataset* cargado de la ruta especificada como primer parámetro para poder estimar el rango de las entradas y las salidas del modelo y que la cuantización de estos valores sea correcta. Así, se carga el modelo de *keras* guardado en la dirección del segundo argumento y se convierte a *Tensorflow Lite* cuantizando los valores de sus parámetros, guardando el modelo resultante en el fichero especificado como tercer argumento. Por ejemplo, podemos crear una versión cuantizada del modelo 3 ejecutando:
```
$ python quantizer.py datasets/dataset_model3.py exported_models/model3.h5 exported_models/model3_quant.tflite
```
### Scripts de inferencia de modelos de Tensorflow Lite
Detallaremos aquí como se implementan y el modo de uso de los *scripts* [`rollout_coral.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/exported_models/rollout_coral.py) y [`rollout_tflite.py`](https://github.com/javigm98/Mejorando-el-Aprendizaje-Automatico/blob/main/exported_models/rollout_tflite.py) que ejecutan inferencias sobre modelos de `Tensorflow Lite`, bien cuantizados o sin cuantizar sobre el acelerador Google Coral (`rollout_coral.py`) o sobre las CPUs del sistema (`rollout_tflite.py`). La estructura de estos dos *scripts* es la misma, salvo que el primero de ellos al crear el intérprete del modelo de *Tensorflow Lite* establece como delegado la TPU. Además, de la función principal de los *scripts*, estos cuenta con dos funciones auxiliares:
  - `make_interpreter(model_file)`. Recibe como parámetro la ruta a un modelo guardado de *Tensorflow Lite* y devuelve un objeto de la clase `Interpreter` sobre el que podremos ejecutar inferencias. En el caso del *script* para la TPU, aquí se indica mediante un delegado que las ejecuciones se realizarán en este soporte.
  - `keep_gping(steps, num_steps, episodes, num_episodes)`: Función que implementa la condición del bucle, indicando cuando debemos parar de ejecutar pasos de inferencia. Se toma directamente del *script* de inferencia que nos proporciona *RLlib* (`rollout.py`).

Cuando ejecutemos el \textit{script} podemos dar valor a una serie de parámetros que configuran las inferencias a realizar:
  - `-m, --model`: ruta al archivo `.tflite` en el que se encuentra el modelo de `Tensorflow Lite` (cuantizado o no) sobre el que ejecutaremos las inferencias.
  - `-s, --steps`: pasos de inferencia que queremos ejecutar. Si damos valor a `--episodes` el número de pasos indicado no tendrá efecto.
  - `-e, --episodes`: número de epsiodios completos de inferencia a ejecutar. Si indicamos su valor, el de `--steps` queda sin efecto.
  - `-o, --output`: ruta a un archivo `.csv` en el que guardaremos los datos relativos a la ejecución de las inferencias (tiempos, pasos por episodio, recompensas...).


Cuando ejecutamos cualesquiera de los dos *scripts* en primer lugar se crea el intérprete para el modelo de *Tensorflow Lite* indicdo. Seguidamente se crea un entorno con `wrap_deepmind` con `Pong-v0` como base, y de aquí será de donde se toman las iteraciones. Ahora, se itera mientras no hayamos completado el número total de episodios (o mientras no hayamos completado el número total de pasos en caso de no haber indicado un número de episodios a ejecutar) y en cada paso de iteración se toma una imagen del entorno, se coloca como tensor de entrada del intérprete del modelo, se invoca al modelo y se obtiene el valor del tensor de salida. De la salida de la política, se toma el índice con el valor más alto y esa será la siguiente acción, que se realiza sobre el entorno, obteniéndose así una nueva observación y comenzando nuevamente el proceso (básicamente es la misma idea que se sigue en el *script* `rollout.py` de *RLlib*.
