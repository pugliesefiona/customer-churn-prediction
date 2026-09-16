import pandas as pd
from pathlib import Path


# Rutas de los archivos
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_ORIGINALES = (
    BASE_DIR
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DATOS_PROCESADOS = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_clean.csv"
)


def cargar_datos():
    """Carga el dataset original de Telco Customer Churn."""
    datos = pd.read_csv(DATOS_ORIGINALES)
    return datos


def limpiar_datos(datos):
    """Limpia y prepara el dataset."""

    # Eliminar espacios al inicio y al final de los nombres de las columnas
    datos.columns = datos.columns.str.strip()

    # Convertir TotalCharges a valores numéricos
    datos["TotalCharges"] = pd.to_numeric(
        datos["TotalCharges"],
        errors="coerce"
    )

    # Eliminar filas con valores faltantes
    datos = datos.dropna()

    # Eliminar registros de clientes duplicados
    datos = datos.drop_duplicates()

    return datos


def main():
    datos = cargar_datos()

    print("Dataset original:")
    print(f"Filas: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    print("\nValores faltantes:")
    print(datos.isnull().sum())

    datos_limpios = limpiar_datos(datos)

    print("\nDataset limpio:")
    print(f"Filas: {datos_limpios.shape[0]}")
    print(f"Columnas: {datos_limpios.shape[1]}")

    DATOS_PROCESADOS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    datos_limpios.to_csv(
        DATOS_PROCESADOS,
        index=False
    )

    print(f"\nDataset limpio guardado en:")
    print(DATOS_PROCESADOS)


if __name__ == "__main__":
    main()