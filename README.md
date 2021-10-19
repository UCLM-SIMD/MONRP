# MONRP
 
Para instalar las dependencias, ejecute ```pip install -r requirements.txt```

## Estructura del repositorio

- ```/algorithms``` contiene los algoritmos utilizados, con su clase principal y sus funciones auxiliares
- ```/datasets``` contiene los datasets utilizados, parseados para ser directamente utilizados por los algoritmos
- ```/evaluation``` métodos auxiliares para evaluar de manera uniforme los resultados de los algoritmos
- ```/galgo``` contiene scripts y configuraciones auxiliares de la experimentación
- ```/models``` define los modelos para genes, individuos, población y problema
- ```/output``` contiene el resultado de la experimentación realizada
- ```executer_driver.py```,```executer_driver_pareto.py```,```execute*.sh```,```galgo_*.sh``` son scripts necesariamente ubicados en el directorio raíz, utilizados para la experimentación en un cluster con colas PBS 
- ```experimentation_analysis.ipynb``` es un notebook de jupyter que muestra el proceso seguido para encontrar las mejores configuraciones de cada algoritmo, calcular las métricas medias y los frente Pareto.

