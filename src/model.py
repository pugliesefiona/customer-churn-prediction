import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# ruta del dataset
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATOS_PROCESADOS = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_clean.csv"
)


def cargar_datos():
    """carga el dataset limpio."""

    datos = pd.read_csv(DATOS_PROCESADOS)

    return datos


def preparar_datos(datos):
    """
    separa las variables predictoras (x) de la variable
    objetivo (y).
    """

    # customerid es solamente un identificador y no aporta
    # información útil para predecir el abandono.
    datos = datos.drop(columns=["customerID"])

    # x contiene las características utilizadas por los modelos.
    X = datos.drop(columns=["Churn"])

    # y contiene la variable que queremos predecir:
    # 0 = el cliente no abandonó
    # 1 = el cliente abandonó.
    y = datos["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


def crear_preprocesador(X):
    """
    define las transformaciones necesarias antes de entrenar
    los modelos.
    """

    # identificamos automáticamente las variables numéricas
    # y categóricas del dataset.
    variables_numericas = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    variables_categoricas = X.select_dtypes(
        include=["object"]
    ).columns

    # para las variables numéricas:
    # 1. reemplazamos posibles valores faltantes por la mediana.
    # 2. estandarizamos los valores para que tengan una escala
    #    comparable.
    transformacion_numerica = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler())
        ]
    )

    # para las variables categóricas:
    # 1. reemplazamos posibles valores faltantes por la categoría
    #    más frecuente.
    # 2. convertimos las categorías en variables numéricas
    #    mediante one-hot encoding.
    transformacion_categorica = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            (
                "codificacion",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # aplicamos cada transformación al tipo de variable
    # correspondiente.
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                transformacion_numerica,
                variables_numericas
            ),
            (
                "categoricas",
                transformacion_categorica,
                variables_categoricas
            )
        ]
    )

    return preprocesador


def evaluar_modelo(
    nombre,
    modelo,
    X_entrenamiento,
    X_prueba,
    y_entrenamiento,
    y_prueba
):
    """
    entrena un modelo y calcula sus principales métricas
    de clasificación.
    """

    # entrenamos el modelo utilizando solamente los datos
    # destinados al entrenamiento.
    modelo.fit(
        X_entrenamiento,
        y_entrenamiento
    )

    # generamos las predicciones sobre los datos de prueba.
    predicciones = modelo.predict(X_prueba)

    # obtenemos la probabilidad estimada de que cada cliente
    # abandone el servicio.
    probabilidades = modelo.predict_proba(X_prueba)[:, 1]

    # accuracy: proporción total de predicciones correctas.
    accuracy = accuracy_score(
        y_prueba,
        predicciones
    )

    # precision: de los clientes que el modelo predijo como
    # churn, qué proporción realmente abandonó.
    precision = precision_score(
        y_prueba,
        predicciones
    )

    # recall: de todos los clientes que realmente abandonaron,
    # qué proporción logró detectar el modelo.
    recall = recall_score(
        y_prueba,
        predicciones
    )

    # f1-score: combina precision y recall en una única métrica.
    f1 = f1_score(
        y_prueba,
        predicciones
    )

    # roc-auc: mide la capacidad del modelo para distinguir
    # entre clientes que abandonan y clientes que permanecen.
    roc_auc = roc_auc_score(
        y_prueba,
        probabilidades
    )

    print(f"\n{nombre}")
    print("-" * len(nombre))
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1-score:  {f1:.4f}")
    print(f"roc-auc:   {roc_auc:.4f}")

    # calculamos la matriz de confusión.
    matriz = confusion_matrix(
        y_prueba,
        predicciones
    )

    print("\nmatriz de confusión:")
    print(matriz)

    # mostramos la matriz de confusión gráficamente.
    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["no", "yes"]
    )

    visualizacion.plot()
    plt.title(f"matriz de confusión - {nombre}")
    plt.tight_layout()
    plt.show()

    # devolvemos las métricas para poder compararlas
    # posteriormente entre los distintos modelos.
    resultados = {
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "roc-auc": roc_auc
    }

    return resultados


def crear_comparacion(resultados):
    """crea un gráfico comparativo de las métricas."""

    resultados_df = pd.DataFrame(resultados)

    metricas = [
        "accuracy",
        "precision",
        "recall",
        "f1-score",
        "roc-auc"
    ]

    resultados_grafico = resultados_df.set_index(
        "modelo"
    )[metricas]

    resultados_grafico.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("comparación de métricas de los modelos")
    plt.xlabel("modelo")
    plt.ylabel("valor")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="métrica")
    plt.tight_layout()
    plt.show()


def main():

    # cargamos el dataset limpio.
    datos = cargar_datos()

    # separamos las variables predictoras de la variable objetivo.
    X, y = preparar_datos(datos)

    print("datos preparados para el modelado:")
    print(f"cantidad de registros: {X.shape[0]}")
    print(f"cantidad de variables: {X.shape[1]}")

    # dividimos los datos en:
    # 80% para entrenamiento
    # 20% para prueba.
    #
    # stratify=y mantiene aproximadamente la misma proporción
    # de churn en ambos conjuntos.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    # creamos el preprocesador que se utilizará antes de
    # entrenar cada modelo.
    preprocesador = crear_preprocesador(X)

    # definimos los tres modelos que vamos a comparar.
    modelos = {

        # modelo lineal utilizado como punto de referencia.
        "regresión logística": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        # modelo basado en reglas de decisión.
        "árbol de decisión": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        # conjunto de múltiples árboles de decisión.
        "random forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    }

    # almacenamos los resultados de cada modelo.
    resultados = []

    # entrenamos y evaluamos cada modelo.
    for nombre, modelo in modelos.items():

        # pipeline:
        # 1. preprocesamiento de los datos.
        # 2. entrenamiento del modelo.
        #
        # esto evita realizar transformaciones sobre todo el
        # dataset antes de dividirlo y reduce el riesgo de
        # data leakage.
        pipeline = Pipeline(
            steps=[
                ("preprocesamiento", preprocesador),
                ("modelo", modelo)
            ]
        )

        resultado = evaluar_modelo(
            nombre,
            pipeline,
            X_entrenamiento,
            X_prueba,
            y_entrenamiento,
            y_prueba
        )

        resultados.append(resultado)

    # mostramos una comparación visual de las métricas
    # obtenidas por los tres modelos.
    crear_comparacion(resultados)


if __name__ == "__main__":
    main()