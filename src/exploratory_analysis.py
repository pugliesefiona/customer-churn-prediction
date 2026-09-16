import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Rutas de los archivos
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_PROCESADOS = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"


def cargar_datos():
    """Carga el dataset de churn previamente limpiado."""
    datos = pd.read_csv(DATOS_PROCESADOS)
    return datos


def analizar_churn(datos):
    """Analiza la distribución general de clientes que abandonaron el servicio."""

    cantidades_churn = datos["Churn"].value_counts()
    porcentaje_churn = datos["Churn"].value_counts(normalize=True) * 100

    print("\nDistribución de churn:")
    print(cantidades_churn)

    print("\nPorcentaje de churn:")
    print(porcentaje_churn.round(2))


def analizar_variables_categoricas(datos):
    """Analiza el churn según diferentes variables categóricas."""

    variables = [
        "Contract",
        "InternetService",
        "PaymentMethod"
    ]

    for columna in variables:
        print(f"\nPorcentaje de churn según {columna}:")

        porcentaje_churn = (
            datos.groupby(columna)["Churn"]
            .apply(lambda grupo: (grupo == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        print(porcentaje_churn.round(2))


def analizar_variables_numericas(datos):
    """Compara las variables numéricas entre clientes con y sin churn."""

    variables = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    print("\nPromedio de variables numéricas según churn:")

    resumen = datos.groupby("Churn")[variables].mean()

    print(resumen.round(2))


def crear_visualizaciones(datos):
    """Crea gráficos para explorar los datos."""

    # Distribución de churn
    plt.figure(figsize=(6, 4))
    sns.countplot(data=datos, x="Churn")
    plt.title("Distribución de clientes según churn")
    plt.xlabel("Churn")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.show()

    # Churn según tipo de contrato
    plt.figure(figsize=(8, 5))
    sns.countplot(data=datos, x="Contract", hue="Churn")
    plt.title("Churn según tipo de contrato")
    plt.xlabel("Tipo de contrato")
    plt.ylabel("Cantidad de clientes")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # Cargos mensuales según churn
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=datos, x="Churn", y="MonthlyCharges")
    plt.title("Cargos mensuales según churn")
    plt.xlabel("Churn")
    plt.ylabel("Cargos mensuales")
    plt.tight_layout()
    plt.show()


def main():
    datos = cargar_datos()

    print("Dataset:")
    print(f"Filas: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    analizar_churn(datos)
    analizar_variables_categoricas(datos)
    analizar_variables_numericas(datos)
    crear_visualizaciones(datos)


if __name__ == "__main__":
    main()