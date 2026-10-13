"""
Descarga indicadores desde WHO GHO API y World Bank API.
Exporta CSVs crudos a data/raw/.

Uso:
    python src/collect_data.py
"""

import time
import requests
import wbgapi as wb
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ANIOS = range(2000, 2023)

# ---------------------------------------------------------------------------
# WHO GHO
# ---------------------------------------------------------------------------

WHO_BASE = "https://ghoapi.azureedge.net/api"

WHO_INDICADORES = {
    "WHOSIS_000001": "esperanza_vida",
    "MDG_0000000007": "mortalidad_infantil",
    "WHOSIS_000004": "mortalidad_adulta",
    "WHS4_100": "vacunacion_dtp",
    "WHS4_117": "vacunacion_hepatitis_b",
    "HWF_0001": "medicos_por_10000",
    "NUTRITION_ANAEMIA_CHILDREN_PREV": "prevalencia_anemia_ninos",
}


def descargar_who(codigo: str, nombre: str) -> pd.DataFrame:
    url = (
        f"{WHO_BASE}/{codigo}"
        f"?$filter=SpatialDimType eq 'COUNTRY'"
        f" and TimeDim ge 2000 and TimeDim le 2022"
    )
    print(f"  WHO -> {nombre} ({codigo})...", end=" ")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    datos = resp.json().get("value", [])

    if not datos:
        print("sin datos")
        return pd.DataFrame()

    df = pd.DataFrame(datos)

    # Filtrar por ambos sexos cuando existe la dimension de sexo.
    # La API cambio el codigo de "BTSX" a "SEX_BTSX"; sin filtro se promedian
    # hombres, mujeres y (en anemia) niveles de severidad.
    if "Dim1" in df.columns:
        btsx = df[df["Dim1"].isin(["BTSX", "SEX_BTSX"])]
        if not btsx.empty:
            df = btsx

    df = df[["SpatialDim", "TimeDim", "NumericValue"]].copy()
    df.columns = ["iso3", "anio", nombre]
    df["anio"] = df["anio"].astype(int)
    df = df.dropna(subset=[nombre])

    # Un valor por pais-anio (promedio si persisten multiples filas)
    df = df.groupby(["iso3", "anio"], as_index=False)[nombre].mean()

    print(f"{len(df)} registros")
    return df


def obtener_who() -> pd.DataFrame:
    print("\n[WHO GHO] Descargando indicadores...")
    frames = []
    for codigo, nombre in WHO_INDICADORES.items():
        df = descargar_who(codigo, nombre)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No se obtuvieron datos de WHO.")

    base = frames[0]
    for df in frames[1:]:
        base = base.merge(df, on=["iso3", "anio"], how="outer")

    base = base.sort_values(["iso3", "anio"]).reset_index(drop=True)
    salida = RAW_DIR / "who_indicators.csv"
    base.to_csv(salida, index=False)
    print(f"  Guardado: {salida}  ({len(base)} filas)")
    return base


# ---------------------------------------------------------------------------
# World Bank
# ---------------------------------------------------------------------------

WB_INDICADORES = {
    "NY.GDP.PCAP.CD": "pib_per_capita",
    "SH.XPD.CHEX.GD.ZS": "gasto_salud_pct_pib",
    "SE.XPD.TOTL.GD.ZS": "gasto_educacion_pct_pib",
    "SH.H2O.BASW.ZS": "acceso_agua_potable",
    "SH.STA.BASS.ZS": "acceso_saneamiento",
    "EG.ELC.ACCS.ZS": "acceso_electricidad",
    "SL.UEM.TOTL.ZS": "tasa_desempleo",
    "SP.URB.TOTL.IN.ZS": "urbanizacion_pct",
    "IT.NET.USER.ZS": "usuarios_internet_pct",
    "SH.DYN.AIDS.ZS": "prevalencia_vih",
}


def descargar_wb(codigo: str, intentos: int = 4) -> list[dict]:
    # La API del World Bank devuelve 504 de forma intermitente en descargas largas
    for intento in range(1, intentos + 1):
        try:
            return [item for item in wb.data.fetch(codigo, time=ANIOS) if item.get("value") is not None]
        except wb.APIError as error:
            if intento == intentos:
                raise
            espera = 15 * intento
            print(f"error {error.code}, reintento en {espera}s...", end=" ")
            time.sleep(espera)


def obtener_world_bank() -> pd.DataFrame:
    print("\n[World Bank] Descargando indicadores...")

    registros = []
    for codigo, nombre in WB_INDICADORES.items():
        print(f"  WB  -> {nombre} ({codigo})...", end=" ")
        items = descargar_wb(codigo)
        registros += [
            {
                "iso3": item["economy"],
                "anio": int(str(item["time"]).replace("YR", "")),
                "serie": item["series"],
                "valor": item["value"],
            }
            for item in items
        ]
        print(f"{len(items)} registros")

    df_long = pd.DataFrame(registros)
    print(f"  Total: {len(df_long)} registros en formato largo")

    df_wide = df_long.pivot_table(
        index=["iso3", "anio"],
        columns="serie",
        values="valor",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns=WB_INDICADORES)

    cols_finales = ["iso3", "anio"] + [c for c in WB_INDICADORES.values() if c in df_wide.columns]
    df_wide = df_wide[cols_finales].sort_values(["iso3", "anio"]).reset_index(drop=True)

    salida = RAW_DIR / "wb_indicators.csv"
    df_wide.to_csv(salida, index=False)
    print(f"  Guardado: {salida}  ({len(df_wide)} filas)")
    return df_wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DESCARGA DE DATOS - Life Expectancy ML")
    print("=" * 60)

    who_df = obtener_who()
    wb_df = obtener_world_bank()

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"WHO   -> {len(who_df):>6} filas | {len(who_df.columns)} columnas")
    print(f"WB    -> {len(wb_df):>6} filas | {len(wb_df.columns)} columnas")
    print("\nListo. Ejecuta src/preprocess.py para continuar.")
