# Mejorando el Aprendizaje Automatico de juegos mediante aceleradores hardware
## Trabajo de Fin de Grado
### Javier Guzmán Muñoz
### Doble Grado en Ingeniería Informática y Matemáticas 
### Universidad Complutense de Madrid

El Aprendizaje por Refuerzo es una rama de la Inteligencia Artificial que aboga por aprender políticas que, mediante la aplicación de acciones, permitan maximizar la recompensa obtenida en un entorno controlado. Las técnicas de Aprendizaje por Refuerzo Profundo se basan en la utilización de redes neuronales de convolución para, a través de representaciones gráficas del entorno, aprender cuáles son las mejores acciones a tomar para acercarse a una situación exitosa. Este tipo de algoritmos se aplican, preferentemente, en ámbitos como la robótica (movimiento autónomo de robots) o el aprendizaje automático de juegos.

El entorno OpenAI GYM (https://gym.openai.com/) es un toolkit muy extendido para comparar el rendimiento de algoritmos de aprendizaje por refuerzo. GYM ofrece entornos que modelan juegos reales (por ejemplo, Pong o Pinball) a través de capturas de pantalla, y permiten observar la calidad del aprendizaje de los algoritmos.

Sin embargo, los algoritmos típicos de Aprendizaje por Refuerzo Profundo son caros desde el punto de vista computacional. El presente TFG propone utilizar aceleradores hardware específicamente diseñados para acelerar el procesamiento de redes neuronales (por ejemplo, Google Coral o Intel NCS), y evaluar su rendimiento en entornos de aprendizaje de juegos reales.

El objetivo final es tanto familiarizarse con el entorno GYM, como realizar un estudio detallado de la ganancia, en términos de tiempo de aprendizaje y aplicación de las políticas aprendidas, en entornos que proponen el aprendizaje autónomo de las reglas de un determinado juego (o juegos), cuando se acelera mediante hardware de propósito específico.
