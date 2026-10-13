"""
Dashboard interactivo — Life Expectancy ML
Streamlit app: prediccion por pais, mapa mundial, comparativa regional y metricas del modelo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Configuracion ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Life Expectancy ML",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent.parent
DATA_PATH  = BASE / "data" / "processed" / "dataset_final.csv"
MODEL_PATH = BASE / "models" / "best_model.pkl"

# Metricas del modelo (split temporal train<=2017 / test>=2018)
MODEL_METRICS = {"R²": 0.9865, "RMSE": 0.838, "MAE": 0.609}

REGION_COLORS = {
    "Africa":   "#e07b54",
    "Americas": "#5b8ff9",
    "Asia":     "#5ad8a6",
    "Europe":   "#f6bd16",
    "Oceania":  "#945fb9",
    "Otro":     "#6e6e6e",
}

# ── Carga de datos y modelo ───────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


# ── Transformacion de features ────────────────────────────────────────────────

def preparar_fila(df_fila: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    df = df_fila.copy()
    df["log_pib_per_capita"] = np.log1p(df["pib_per_capita"])
    df = df.drop(columns=["pib_per_capita"], errors="ignore")

    region_dummies = pd.get_dummies(df["region"], prefix="region")
    df = pd.concat([df, region_dummies], axis=1)

    drop_cols = ["iso3", "pais", "region", "esperanza_vida"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ── Graficos auxiliares ───────────────────────────────────────────────────────

def shap_waterfall_plotly(shap_vals, feature_names, base_value, prediction, top_n=10):
    """Genera un grafico waterfall de SHAP con Plotly."""
    pairs = sorted(zip(shap_vals, feature_names), key=lambda x: abs(x[0]), reverse=True)
    vals, names = zip(*pairs[:top_n])

    # Acumulacion para el waterfall
    running = base_value
    x_starts, x_widths, colors, labels = [], [], [], []
    for v, n in zip(vals, names):
        x_starts.append(running)
        x_widths.append(v)
        colors.append("#e07b54" if v < 0 else "#5b8ff9")
        labels.append(f"{n}<br>{v:+.3f}")
        running += v

    fig = go.Figure()

    # Barras SHAP
    for i, (start, width, color, label) in enumerate(zip(x_starts, x_widths, colors, labels)):
        fig.add_trace(go.Bar(
            x=[width],
            y=[names[i]],
            base=[start],
            orientation="h",
            marker_color=color,
            text=f"{vals[i]:+.3f}",
            textposition="outside",
            hovertemplate=f"<b>{names[i]}</b><br>Impacto SHAP: {vals[i]:+.3f}<extra></extra>",
            showlegend=False,
        ))

    # Linea de valor base
    fig.add_vline(x=base_value, line_dash="dot", line_color="gray",
                  annotation_text=f"Base: {base_value:.1f}", annotation_position="top")
    # Linea de prediccion
    fig.add_vline(x=prediction, line_dash="dash", line_color="#5ad8a6",
                  annotation_text=f"Pred: {prediction:.1f}", annotation_position="top right")

    fig.update_layout(
        title="Factores que influyen en la prediccion (SHAP)",
        xaxis_title="Esperanza de vida (anos)",
        yaxis_title="",
        height=420,
        margin=dict(l=20, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(autorange="reversed")
    return fig


# ── App principal ─────────────────────────────────────────────────────────────

def main():
    df = load_data()
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    explainer = load_explainer(model)

    paises = sorted(df["pais"].unique())
    anios = sorted(df["anio"].unique())

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🌍 Life Expectancy ML")
        st.markdown("Modelo predictivo de esperanza de vida · XGBoost · R²=0.9865")
        st.divider()

        st.subheader("Seleccion")
        pais_sel = st.selectbox("Pais", paises, index=paises.index("Chile"))
        anio_sel = st.select_slider("Ano", options=anios, value=2021)

        st.divider()
        st.caption("Datos: OMS (WHO) y Banco Mundial · Periodo 2000-2021 · 185 paises")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Prediccion", "🌍 Mapa Mundial", "📊 Comparativa", "🤖 Metricas del Modelo"]
    )

    # ── TAB 1: Prediccion ────────────────────────────────────────────────────
    with tab1:
        fila = df[(df["pais"] == pais_sel) & (df["anio"] == anio_sel)]

        if fila.empty:
            st.warning(f"No hay datos para {pais_sel} en {anio_sel}.")
        else:
            row = fila.iloc[[0]]
            real = row["esperanza_vida"].values[0]
            X_row = preparar_fila(row, feature_names)
            pred = float(model.predict(X_row)[0])
            error = abs(pred - real)

            st.subheader(f"{pais_sel} · {anio_sel}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediccion del modelo", f"{pred:.2f} anos")
            col2.metric("Valor real", f"{real:.2f} anos")
            col3.metric("Error absoluto", f"{error:.2f} anos",
                        delta=f"{'Sobreestima' if pred > real else 'Subestima'} por {error:.2f}",
                        delta_color="inverse")

            st.divider()

            # SHAP waterfall
            shap_values = explainer(X_row)
            sv = shap_values.values[0]
            base = float(shap_values.base_values[0])
            fig_shap = shap_waterfall_plotly(sv, feature_names, base, pred)
            st.plotly_chart(fig_shap, use_container_width=True)

            # Indicadores clave del pais
            with st.expander("Ver indicadores del pais"):
                indicadores = {
                    "Mortalidad infantil":     row["mortalidad_infantil"].values[0],
                    "Mortalidad adulta":        row["mortalidad_adulta"].values[0],
                    "PIB per capita (USD)":     row["pib_per_capita"].values[0],
                    "Acceso electricidad (%)":  row["acceso_electricidad"].values[0],
                    "Acceso saneamiento (%)":   row["acceso_saneamiento"].values[0],
                    "Acceso agua potable (%)":  row["acceso_agua_potable"].values[0],
                    "Gasto en salud (% PIB)":   row["gasto_salud_pct_pib"].values[0],
                    "Prevalencia VIH":          row["prevalencia_vih"].values[0],
                }
                st.dataframe(
                    pd.DataFrame(list(indicadores.items()), columns=["Indicador", "Valor"])
                    .set_index("Indicador")
                    .style.format({"Valor": "{:.2f}"}),
                    use_container_width=True,
                )

    # ── TAB 2: Mapa Mundial ──────────────────────────────────────────────────
    with tab2:
        anio_mapa = st.select_slider("Ano del mapa", options=anios, value=2021, key="mapa_anio")
        df_mapa = df[df["anio"] == anio_mapa]

        fig_mapa = px.choropleth(
            df_mapa,
            locations="iso3",
            color="esperanza_vida",
            hover_name="pais",
            hover_data={"iso3": False, "esperanza_vida": ":.1f", "region": True},
            color_continuous_scale="RdYlGn",
            range_color=[45, 85],
            labels={"esperanza_vida": "Esperanza de vida"},
            title=f"Esperanza de vida al nacer · {anio_mapa}",
        )
        fig_mapa.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(showframe=False, showcoastlines=True, bgcolor="rgba(0,0,0,0)"),
            coloraxis_colorbar=dict(title="Anos"),
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Media global", f"{df_mapa['esperanza_vida'].mean():.1f} anos")
        col_b.metric("Maximo", f"{df_mapa['esperanza_vida'].max():.1f} anos  ({df_mapa.loc[df_mapa['esperanza_vida'].idxmax(), 'pais']})")
        col_c.metric("Minimo", f"{df_mapa['esperanza_vida'].min():.1f} anos  ({df_mapa.loc[df_mapa['esperanza_vida'].idxmin(), 'pais']})")

    # ── TAB 3: Comparativa ───────────────────────────────────────────────────
    with tab3:
        regiones_disp = sorted(df["region"].unique())
        regiones_sel = st.multiselect(
            "Comparar con regiones",
            options=regiones_disp,
            default=["Americas", "Europe"],
        )

        # Serie de Chile
        chile = df[df["pais"] == "Chile"].groupby("anio")["esperanza_vida"].mean().reset_index()
        chile["serie"] = "Chile"

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=chile["anio"], y=chile["esperanza_vida"],
            mode="lines+markers", name="Chile",
            line=dict(color="#f6bd16", width=3),
            marker=dict(size=6),
        ))

        for region in regiones_sel:
            media_reg = df[df["region"] == region].groupby("anio")["esperanza_vida"].mean().reset_index()
            fig_comp.add_trace(go.Scatter(
                x=media_reg["anio"], y=media_reg["esperanza_vida"],
                mode="lines", name=f"{region} (media)",
                line=dict(color=REGION_COLORS.get(region, "#aaa"), width=2, dash="dot"),
            ))

        fig_comp.update_layout(
            title="Evolucion de la esperanza de vida · Chile vs. regiones (2000-2021)",
            xaxis_title="Ano",
            yaxis_title="Esperanza de vida (anos)",
            height=440,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_comp.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_comp.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Tabla comparativa ultimo anio disponible
        ultimo = df["anio"].max()
        st.subheader(f"Comparativa de indicadores clave · {ultimo}")
        cols_tabla = ["pais", "region", "esperanza_vida", "mortalidad_infantil",
                      "pib_per_capita", "acceso_electricidad", "acceso_saneamiento"]
        df_tabla = df[(df["anio"] == ultimo) & (
            (df["pais"] == "Chile") | (df["region"].isin(regiones_sel))
        )][cols_tabla].sort_values("esperanza_vida", ascending=False)
        st.dataframe(df_tabla.set_index("pais").style.format({
            "esperanza_vida": "{:.1f}",
            "mortalidad_infantil": "{:.1f}",
            "pib_per_capita": "{:,.0f}",
            "acceso_electricidad": "{:.1f}",
            "acceso_saneamiento": "{:.1f}",
        }), use_container_width=True)

    # ── TAB 4: Metricas del Modelo ───────────────────────────────────────────
    with tab4:
        st.subheader("Rendimiento del modelo · Split temporal (train<=2017 / test>=2018)")

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{MODEL_METRICS['R²']:.4f}", help="Coeficiente de determinacion")
        c2.metric("RMSE", f"{MODEL_METRICS['RMSE']:.3f} anos", help="Raiz del error cuadratico medio")
        c3.metric("MAE", f"{MODEL_METRICS['MAE']:.3f} anos", help="Error absoluto medio")

        st.caption("El split temporal es la evaluacion mas rigurosa: entrena en datos historicos y predice anos futuros.")

        st.divider()

        # Feature importance global del modelo
        st.subheader("Importancia de features (XGBoost gain)")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(15)

        fig_fi = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Importancia (gain)", "feature": ""},
            title="Top 15 features por importancia",
        )
        fig_fi.update_layout(
            height=460,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        fig_fi.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig_fi, use_container_width=True)


if __name__ == "__main__":
    main()
