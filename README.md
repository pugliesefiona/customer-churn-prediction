# customer churn prediction

proyecto de machine learning para analizar y predecir el abandono de clientes de una empresa de telecomunicaciones utilizando python y scikit-learn.

## sobre el proyecto

el objetivo es entender qué características están relacionadas con el abandono de clientes y probar distintos modelos de machine learning para predecirlo.

se trabajó con información sobre los clientes, los servicios contratados, el tipo de contrato, el método de pago y los cargos mensuales y totales.

## dataset

se utilizó el dataset **telco customer churn**, que contiene información de 7.043 clientes y 21 variables.

después de limpiar los datos, quedaron **7.032 registros**. Se eliminaron 11 registros porque sus valores de `TotalCharges` no podían convertirse correctamente a números.

## análisis

se exploraron diferentes características de los clientes para identificar patrones relacionados con el churn.

algunos de los patrones observados fueron:

* el 26,58 % de los clientes presentó churn
* los contratos mensuales tuvieron una mayor proporción de churn
* los clientes con fibra óptica presentaron una mayor proporción de churn
* los clientes que abandonaron el servicio tenían una menor antigüedad promedio
* también presentaban cargos mensuales promedio más altos

## modelos

se probaron tres modelos de clasificación:

* regresión logística
* árbol de decisión
* random forest

los datos se dividieron en un 80 % para entrenamiento y un 20 % para prueba.

el preprocesamiento se realizó utilizando `Pipeline` y `ColumnTransformer` de scikit-learn, incluyendo la transformación de variables categóricas y la estandarización de las variables numéricas.

los modelos se evaluaron utilizando accuracy, precision, recall, f1-score y roc-auc, además de matrices de confusión.

## tecnologías

* python
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* git
* github

## cómo ejecutarlo

crear un entorno virtual e instalar las dependencias:


python -m venv .venv
pip install -r requirements.txt


colocar el dataset original dentro de `data/raw/` y ejecutar:

python src/data_cleaning.py
python src/exploratory_analysis.py
python src/feature_engineering.py
python src/model.py
```

## posibles mejoras

* validación cruzada
* optimización de hiperparámetros
* nuevas variables
* técnicas para manejar el desbalance de clases
* interpretación de los modelos
* prueba de otros algoritmos



