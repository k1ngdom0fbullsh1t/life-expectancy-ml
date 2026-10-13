"""
Dashboard interactivo — Life Expectancy ML
Predicción por país con explicación SHAP, mapa mundial, comparativa regional y métricas del modelo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Life Expectancy ML",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent.parent
DATA_PATH  = BASE / "data" / "processed" / "dataset_final.csv"
MODEL_PATH = BASE / "models" / "dashboard_model.pkl"
REPO_URL   = "https://github.com/k1ngdom0fbullsh1t/life-expectancy-ml"

REGION_ES = {
    "Africa":   "África",
    "Americas": "América",
    "Asia":     "Asia",
    "Europe":   "Europa",
    "Oceania":  "Oceanía",
}

OCDE = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD",
    "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
}

FEATURE_LABELS = {
    "anio":                     "Año",
    "mortalidad_infantil":      "Mortalidad infantil",
    "mortalidad_adulta":        "Mortalidad adulta",
    "prevalencia_vih":          "Personas con VIH (estimado)",
    "vacunacion_dtp":           "Vacunación DTP",
    "vacunacion_hepatitis_b":   "Vacunación hepatitis B",
    "medicos_por_10000":        "Médicos por 10.000 hab.",
    "prevalencia_anemia_ninos": "Anemia infantil",
    "log_pib_per_capita":       "PIB per cápita (log)",
    "gasto_salud_pct_pib":      "Gasto en salud (% PIB)",
    "gasto_educacion_pct_pib":  "Gasto en educación (% PIB)",
    "acceso_agua_potable":      "Acceso a agua potable",
    "acceso_saneamiento":       "Acceso a saneamiento",
    "acceso_electricidad":      "Acceso a electricidad",
    "tasa_desempleo":           "Tasa de desempleo",
    "urbanizacion_pct":         "Urbanización",
    "usuarios_internet_pct":    "Usuarios de internet",
}

INDICADORES = {
    "esperanza_vida":      ("Esperanza de vida (años)", "{:.1f}"),
    "mortalidad_infantil": ("Mortalidad infantil (por 1.000)", "{:.1f}"),
    "mortalidad_adulta":   ("Mortalidad adulta (por 1.000)", "{:.1f}"),
    "pib_per_capita":      ("PIB per cápita (USD)", "{:,.0f}"),
    "acceso_electricidad": ("Acceso a electricidad (%)", "{:.1f}"),
    "acceso_saneamiento":  ("Acceso a saneamiento (%)", "{:.1f}"),
    "acceso_agua_potable": ("Acceso a agua potable (%)", "{:.1f}"),
    "gasto_salud_pct_pib": ("Gasto en salud (% PIB)", "{:.1f}"),
    "prevalencia_vih":     ("Personas con VIH (estimado)", "{:,.0f}"),
}

AZUL, ROJO, VERDE, AMARILLO = "#5b8ff9", "#e07b54", "#5ad8a6", "#f6bd16"
GRID = "rgba(128,128,128,0.2)"
COLORES_REGION = {"África": ROJO, "América": AZUL, "Asia": VERDE, "Europa": AMARILLO, "Oceanía": "#945fb9"}
LEYENDA_ABAJO =dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title_text="")


def etiqueta(feature: str) -> str:
    if feature.startswith("region_"):
        region = feature.removeprefix("region_")
        return f"Región: {REGION_ES.get(region, region)}"
    return FEATURE_LABELS.get(feature, feature)


def layout_base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


# ── Carga ─────────────────────────────────────────────────────────────────────

# La fecha de modificacion va en la clave de cache: Streamlit Cloud recarga el codigo sin
# reiniciar el proceso, y sin ella se reutilizan datos o modelos de una version anterior.
def version(path: Path) -> float:
    return path.stat().st_mtime


@st.cache_data
def load_data(data_version: float) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["region_es"] = df["region"].map(REGION_ES)
    return df


@st.cache_resource
def load_model(model_version: float) -> tuple[dict, shap.TreeExplainer]:
    artifact = joblib.load(MODEL_PATH)
    return artifact, shap.TreeExplainer(artifact["model"])


XGB_DTYPES = {"i": bool, "int": "int64", "float": "float64"}


def preparar_X(df: pd.DataFrame, model) -> pd.DataFrame:
    booster = model.get_booster()
    X = df.copy()
    X["log_pib_per_capita"] = np.log1p(X["pib_per_capita"])
    X = pd.concat([X, pd.get_dummies(X["region"], prefix="region")], axis=1)
    X = X.reindex(columns=booster.feature_names, fill_value=0)
    return X.astype({c: XGB_DTYPES[t] for c, t in zip(booster.feature_names, booster.feature_types)})


@st.cache_data
def importancia_shap(_explainer, X: pd.DataFrame, model_version: float) -> pd.Series:
    valores = _explainer(X).values
    return pd.Series(np.abs(valores).mean(axis=0), index=X.columns)


def grupos_comparacion(df: pd.DataFrame) -> dict:
    grupos = {
        "LATAM": (df["region"] == "Americas") & ~df["iso3"].isin(["USA", "CAN"]),
        "OCDE":  df["iso3"].isin(OCDE),
    }
    for region, nombre in REGION_ES.items():
        grupos[nombre] = df["region"] == region
    return grupos


# ── Gráficos ──────────────────────────────────────────────────────────────────

def shap_waterfall(shap_vals, feature_names, base_value, prediction, top_n=8) -> go.Figure:
    orden = np.argsort(-np.abs(shap_vals))
    top, resto = orden[:top_n], orden[top_n:]

    nombres = [etiqueta(feature_names[i]) for i in top]
    valores = [float(shap_vals[i]) for i in top]
    if len(resto):
        nombres.append(f"Otras {len(resto)} variables")
        valores.append(float(shap_vals[resto].sum()))

    acumulado = base_value + np.cumsum(valores)
    lo = min(base_value, acumulado.min())
    hi = max(base_value, acumulado.max())
    pad = 0.25 * (hi - lo) + 0.5

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(valores) + ["total"],
        y=["Promedio del modelo"] + nombres + ["Predicción"],
        x=[base_value] + valores + [0],
        text=[f"{base_value:.2f}"] + [f"{v:+.2f}" for v in valores] + [f"{prediction:.2f}"],
        textposition="outside",
        increasing=dict(marker=dict(color=AZUL)),
        decreasing=dict(marker=dict(color=ROJO)),
        totals=dict(marker=dict(color=VERDE)),
        connector=dict(line=dict(color="rgba(128,128,128,0.4)", dash="dot")),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(title="¿Por qué el modelo predice este valor? (SHAP)",
                      xaxis_title="Esperanza de vida (años)", showlegend=False)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return layout_base(fig, 440)


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    df = load_data(version(DATA_PATH))
    artifact, explainer = load_model(version(MODEL_PATH))
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    train_max_year = artifact["train_max_year"]
    metricas = artifact["metrics"]
    grupos = grupos_comparacion(df)

    paises = sorted(df["pais"].unique())
    anios = sorted(df["anio"].unique())
    anio_min, anio_max = anios[0], anios[-1]

    with st.sidebar:
        st.title("🌍 Life Expectancy ML")
        st.markdown(
            f"Modelo predictivo de esperanza de vida · XGBoost · "
            f"R² {metricas['r2']:.4f} en {train_max_year + 1}–{anio_max} (fuera de muestra)"
        )
        st.divider()

        st.subheader("Selección")
        pais_sel = st.selectbox("País", paises, index=paises.index("Chile"))
        anio_sel = st.select_slider("Año", options=anios, value=anio_max)

        st.divider()
        st.caption(
            f"Datos: OMS (WHO) y Banco Mundial · {anio_min}–{anio_max} · "
            f"{df['pais'].nunique()} países"
        )
        st.caption(f"[Código en GitHub]({REPO_URL})")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Predicción", "🌍 Mapa mundial", "📊 Comparativa", "🤖 Métricas del modelo"]
    )

    # ── Predicción ───────────────────────────────────────────────────────────
    with tab1:
        fila = df[(df["pais"] == pais_sel) & (df["anio"] == anio_sel)]

        if fila.empty:
            st.warning(f"No hay datos para {pais_sel} en {anio_sel}.")
        else:
            real = float(fila["esperanza_vida"].iloc[0])
            X_row = preparar_X(fila, model)
            pred = float(model.predict(X_row)[0])
            error = abs(pred - real)

            st.subheader(f"{pais_sel} · {anio_sel}")
            if anio_sel > train_max_year:
                st.success(
                    f"Predicción fuera de muestra: el modelo se entrenó con datos hasta "
                    f"{train_max_year} y nunca vio este año."
                )
            else:
                st.info(
                    f"Este año fue parte del entrenamiento ({anio_min}–{train_max_year}), así que "
                    f"el error es optimista. Elige {train_max_year + 1}–{anio_max} para ver "
                    f"predicciones fuera de muestra."
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicción del modelo", f"{pred:.2f} años")
            c2.metric("Valor real", f"{real:.2f} años")
            c3.metric("Error absoluto", f"{error:.2f} años")
            st.caption(
                f"El modelo {'sobreestima' if pred > real else 'subestima'} "
                f"la esperanza de vida en {error:.2f} años."
            )

            st.divider()

            sv = explainer(X_row)
            fig_shap = shap_waterfall(sv.values[0], feature_names, float(sv.base_values[0]), pred)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(
                "Cada barra muestra cuántos años suma (azul) o resta (rojo) cada variable "
                "a partir del promedio del modelo."
            )

            with st.expander("Ver indicadores del país"):
                tabla = pd.DataFrame(
                    [(nombre, fmt.format(fila[col].iloc[0])) for col, (nombre, fmt) in INDICADORES.items()],
                    columns=["Indicador", "Valor"],
                ).set_index("Indicador")
                st.dataframe(tabla, use_container_width=True)

    # ── Mapa mundial ─────────────────────────────────────────────────────────
    with tab2:
        anio_mapa = st.select_slider("Año del mapa", options=anios, value=anio_max, key="mapa_anio")
        df_mapa = df[df["anio"] == anio_mapa]

        fig_mapa = px.choropleth(
            df_mapa,
            locations="iso3",
            color="esperanza_vida",
            hover_name="pais",
            hover_data={"iso3": False, "esperanza_vida": ":.1f", "region_es": True},
            color_continuous_scale="RdYlGn",
            range_color=[45, 85],
            labels={"esperanza_vida": "Esperanza de vida", "region_es": "Región"},
            title=f"Esperanza de vida al nacer · {anio_mapa}",
        )
        fig_mapa.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(showframe=False, showcoastlines=True, bgcolor="rgba(0,0,0,0)"),
            coloraxis_colorbar=dict(title="Años"),
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

        maximo = df_mapa.loc[df_mapa["esperanza_vida"].idxmax()]
        minimo = df_mapa.loc[df_mapa["esperanza_vida"].idxmin()]
        ca, cb, cc = st.columns(3)
        ca.metric("Promedio mundial", f"{df_mapa['esperanza_vida'].mean():.1f} años")
        cb.metric(f"Máximo · {maximo['pais']}", f"{maximo['esperanza_vida']:.1f} años")
        cc.metric(f"Mínimo · {minimo['pais']}", f"{minimo['esperanza_vida']:.1f} años")

    # ── Comparativa ──────────────────────────────────────────────────────────
    with tab3:
        grupos_sel = st.multiselect(
            f"Comparar {pais_sel} con",
            options=list(grupos),
            default=["LATAM", "OCDE"],
        )

        serie_pais = df[df["pais"] == pais_sel].sort_values("anio")
        fig_comp = go.Figure(go.Scatter(
            x=serie_pais["anio"], y=serie_pais["esperanza_vida"],
            mode="lines+markers", name=pais_sel,
            line=dict(color=AMARILLO, width=3), marker=dict(size=6),
        ))
        paleta = px.colors.qualitative.Set2
        for i, grupo in enumerate(grupos_sel):
            media = df[grupos[grupo]].groupby("anio")["esperanza_vida"].mean()
            fig_comp.add_trace(go.Scatter(
                x=media.index, y=media.values, mode="lines", name=f"{grupo} (promedio)",
                line=dict(color=paleta[i % len(paleta)], width=2, dash="dot"),
            ))
        fig_comp.update_layout(
            title=f"Esperanza de vida · {pais_sel} vs. grupos ({anio_min}–{anio_max})",
            xaxis_title="Año",
            yaxis_title="Esperanza de vida (años)",
            hovermode="x unified",
            legend=LEYENDA_ABAJO,
        )
        layout_base(fig_comp, 480).update_layout(margin=dict(b=110))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("Los promedios de cada grupo son promedios simples entre países (sin ponderar por población).")

        df_ult = df[df["anio"] == anio_max]
        columnas = {pais_sel: df_ult[df_ult["pais"] == pais_sel]}
        for grupo in grupos_sel:
            columnas[f"{grupo} (promedio)"] = df_ult[grupos[grupo].loc[df_ult.index]]
        tabla_comp = pd.DataFrame(
            {
                nombre: [fmt.format(datos[col].mean()) for col, (_, fmt) in INDICADORES.items()]
                for nombre, datos in columnas.items()
            },
            index=[etiqueta_ind for etiqueta_ind, _ in INDICADORES.values()],
        )

        st.subheader(f"Indicadores clave · {anio_max}")
        st.dataframe(tabla_comp, use_container_width=True)

    # ── Métricas del modelo ──────────────────────────────────────────────────
    with tab4:
        st.subheader(f"Rendimiento fuera de muestra · entrenado {anio_min}–{train_max_year}, evaluado {train_max_year + 1}–{anio_max}")

        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metricas['r2']:.4f}", help="Proporción de la varianza explicada por el modelo")
        m2.metric("RMSE", f"{metricas['rmse']:.3f} años", help="Raíz del error cuadrático medio")
        m3.metric("MAE", f"{metricas['mae']:.3f} años", help="Error absoluto medio")
        st.caption(
            "El split temporal es la evaluación más exigente: el modelo predice años que nunca vio, "
            "como ocurriría en un uso real."
        )

        st.divider()

        X_all = preparar_X(df, model)
        imp = importancia_shap(explainer, X_all, version(MODEL_PATH)).sort_values().tail(12)
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=[etiqueta(f) for f in imp.index],
            orientation="h", marker_color=AZUL,
            hovertemplate="%{y}: %{x:.2f} años<extra></extra>",
        ))
        fig_imp.update_layout(title="Importancia global de cada variable (promedio de |SHAP|)",
                              xaxis_title="Impacto promedio en la predicción (años)")
        st.plotly_chart(layout_base(fig_imp, 440), use_container_width=True)

        test = df[df["anio"] > train_max_year].copy()
        test["prediccion"] = model.predict(preparar_X(test, model))
        fig_pr = px.scatter(
            test, x="esperanza_vida", y="prediccion", color="region_es",
            color_discrete_map=COLORES_REGION,
            hover_name="pais", hover_data={"anio": True, "region_es": False},
            labels={"esperanza_vida": "Valor real (años)", "prediccion": "Predicción (años)",
                    "region_es": "Región", "anio": "Año"},
            title=f"Predicción vs. valor real · {train_max_year + 1}–{anio_max}",
            opacity=0.7,
        )
        lim = [test[["esperanza_vida", "prediccion"]].min().min() - 1,
               test[["esperanza_vida", "prediccion"]].max().max() + 1]
        fig_pr.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Predicción perfecta",
                                    line=dict(color="gray", dash="dash")))
        fig_pr.update_layout(legend=LEYENDA_ABAJO)
        layout_base(fig_pr, 540).update_layout(margin=dict(b=110))
        st.plotly_chart(fig_pr, use_container_width=True)


if __name__ == "__main__":
    main()
