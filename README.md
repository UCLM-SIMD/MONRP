# MONRP
 
Para instalar las dependencias, ejecute ```pip install -r requirements.txt```

## Estructura del repositorio

- ```/algorithms``` contiene los algoritmos utilizados, con su clase principal y sus funciones auxiliares
- ```/datasets``` contiene los datasets utilizados, parseados para ser directamente utilizados por los algoritmos
- ```/galgo``` contiene scripts y configuraciones auxiliares de la experimentación
- ```/models``` define los modelos para genes, individuos, población y problema
- ```/output``` contiene el resultado de la experimentación realizada
- ```ejecutor_galgo.py```,```execute.sh```,```executer.py```,```galgo.sh```,```galgo_test.sh``` son scripts necesariamente ubicados en el directorio raíz, utilizados para la experimentación en un cluster con colas PBS 
- ```analyisis.ipynb``` es un notebook de jupyter que muestra el proceso seguido para encontrar las mejores configuraciones de cada algoritmo, calcular las métricas medias y generar los gráficos de Kiviat y de frente Pareto.

## Ejecución
Para ejecutar el notebook y reproducir los pasos es necesario ordenar la ejecución de todos los bloques del notebook.
El script ```problem_instance.py``` puede ser ejecutado para visualizar una comparativa de frentes de Pareto de 3 configuraciones de algoritmos.
