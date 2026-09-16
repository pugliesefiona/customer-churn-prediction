import pandas as pd
from pathlib import Path


# Rutas de los archivos
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_PROCESADOS = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
DATOS_FEATURES = BASE_DIR / "data" / "processed" / "telco_churn_features.csv"


def cargar_datos():
    """Carga el dataset previamente limpiado."""
    datos = pd.read_csv(DATOS_PROCESADOS)
    return datos


def preparar_variables(datos):
    """Prepara las variables para los modelos de machine learning."""

    # El identificador del cliente no aporta información predictiva
    datos = datos.drop(columns=["customerID"])

    # Convertir la variable objetivo Churn a valores numéricos
    datos["Churn"] = datos["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Identificar variables categóricas
    variables_categoricas = datos.select_dtypes(
        include=["object"]
    ).columns

    # Convertir variables categóricas en variables binarias
    datos = pd.get_dummies(
        datos,
        columns=variables_categoricas,
        drop_first=True,
        dtype=int
    )

    return datos


def guardar_datos(datos):
    """Guarda el dataset preparado para el modelado."""

    DATOS_FEATURES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    datos.to_csv(
        DATOS_FEATURES,
        index=False
    )

    print(f"\nDataset preparado guardado en:")
    print(DATOS_FEATURES)


def main():
    datos = cargar_datos()

    print("Dataset original:")
    print(f"Filas: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    datos_preparados = preparar_variables(datos)

    print("\nDataset preparado para machine learning:")
    print(f"Filas: {datos_preparados.shape[0]}")
    print(f"Columnas: {datos_preparados.shape[1]}")

    print("\nPrimeras filas:")
    print(datos_preparados.head())

    guardar_datos(datos_preparados)


if __name__ == "__main__":
    main()