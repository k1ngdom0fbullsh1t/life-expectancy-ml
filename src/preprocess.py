"""
Merge de datasets WHO + World Bank, limpieza e imputacion.
Lee de data/raw/ y exporta data/processed/dataset_final.csv

Uso:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
import pycountry
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REGION_MAP = {
    "Africa": [
        "DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG",
        "CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN",
        "LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA",
        "RWA","STP","SEN","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE",
    ],
    "Asia": [
        "AFG","ARM","AZE","BHR","BGD","BTN","BRN","KHM","CHN","CYP","GEO","IND","IDN",
        "IRN","IRQ","ISR","JPN","JOR","KAZ","KWT","KGZ","LAO","LBN","MYS","MDV","MNG",
        "MMR","NPL","PRK","OMN","PAK","PSE","PHL","QAT","SAU","SGP","KOR","LKA","SYR",
        "TWN","TJK","THA","TLS","TKM","ARE","UZB","VNM","YEM",
    ],
    "Europe": [
        "ALB","AND","AUT","BLR","BEL","BIH","BGR","HRV","CZE","DNK","EST","FIN","FRA",
        "DEU","GRC","HUN","ISL","IRL","ITA","XKX","LVA","LIE","LTU","LUX","MLT","MDA",
        "MCO","MNE","NLD","MKD","NOR","POL","PRT","ROU","RUS","SMR","SRB","SVK","SVN",
        "ESP","SWE","CHE","UKR","GBR","VAT",
    ],
    "Americas": [
        "ATG","ARG","BHS","BRB","BLZ","BOL","BRA","CAN","CHL","COL","CRI","CUB","DMA",
        "DOM","ECU","SLV","GRD","GTM","GUY","HTI","HND","JAM","MEX","NIC","PAN","PRY",
        "PER","KNA","LCA","VCT","SUR","TTO","USA","URY","VEN",
    ],
    "Oceania": [
        "AUS","FJI","KIR","MHL","FSM","NRU","NZL","PLW","PNG","WSM","SLB","TON","TUV","VUT",
    ],
}

ISO3_A_REGION = {iso3: region for region, paises in REGION_MAP.items() for iso3 in paises}


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    who = pd.read_csv(RAW_DIR / "who_indicators.csv")
    wb = pd.read_csv(RAW_DIR / "wb_indicators.csv")
    print(f"WHO cargado:  {len(who):>6} filas")
    print(f"WB  cargado:  {len(wb):>6} filas")
    return who, wb


def validar_iso3(df: pd.DataFrame) -> pd.DataFrame:
    paises_validos = {c.alpha_3 for c in pycountry.countries}
    antes = len(df)
    df = df[df["iso3"].isin(paises_validos)].copy()
    print(f"  Filtro ISO-3: {antes} -> {len(df)} filas ({antes - len(df)} eliminadas)")
    return df


def imputar_por_grupo(df: pd.DataFrame, col: str) -> pd.Series:
    mediana_grupo = df.groupby(["region", "anio"])[col].transform("median")
    mediana_global = df[col].median()
    return df[col].fillna(mediana_grupo).fillna(mediana_global)


def limpiar_y_mergear(who: pd.DataFrame, wb: pd.DataFrame) -> pd.DataFrame:
    print("\n[Preprocesamiento] Merge WHO + World Bank...")

    who = validar_iso3(who)
    wb = validar_iso3(wb)

    df = who.merge(wb, on=["iso3", "anio"], how="inner")
    print(f"  Merge inner: {len(df)} filas | {len(df.columns)} columnas")

    def iso3_a_nombre(iso3):
        try:
            return pycountry.countries.get(alpha_3=iso3).name
        except AttributeError:
            return None

    df["pais"] = df["iso3"].map(iso3_a_nombre)
    df["region"] = df["iso3"].map(ISO3_A_REGION).fillna("Otro")

    antes = len(df)
    df = df.dropna(subset=["esperanza_vida"])
    print(f"  Sin target eliminadas: {antes - len(df)} filas")

    features = [c for c in df.columns if c not in ["iso3", "anio", "pais", "region", "esperanza_vida"]]
    for col in features:
        if df[col].isna().sum() > 0:
            df[col] = imputar_por_grupo(df, col)

    nulos_totales = df[features].isna().sum().sum()
    print(f"  Nulos restantes tras imputacion: {nulos_totales}")

    primeras = ["iso3", "pais", "region", "anio", "esperanza_vida"]
    resto = [c for c in df.columns if c not in primeras]
    df = df[primeras + resto].sort_values(["iso3", "anio"]).reset_index(drop=True)

    return df


def guardar(df: pd.DataFrame) -> None:
    salida = PROCESSED_DIR / "dataset_final.csv"
    df.to_csv(salida, index=False)
    print(f"\n  Guardado: {salida}")
    print(f"  Shape final: {df.shape}")
    print(f"  Paises unicos: {df['iso3'].nunique()}")
    print(f"  Anios: {df['anio'].min()} - {df['anio'].max()}")
    print(f"  Columnas: {list(df.columns)}")


if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESAMIENTO - Life Expectancy ML")
    print("=" * 60)

    who, wb = cargar_datos()
    df = limpiar_y_mergear(who, wb)
    guardar(df)

    print("\nListo. Abre notebooks/02_eda.ipynb para el analisis exploratorio.")
