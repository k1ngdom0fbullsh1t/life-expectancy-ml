"""
Entrena el modelo XGBoost con los hiperparametros optimos (GridSearchCV del notebook 03).

Genera:
    models/dashboard_model.pkl -> modelo entrenado con split temporal (train <= 2017),
                                  usado por el dashboard para que 2018-2021 sean
                                  predicciones fuera de muestra reales.

Uso:
    python src/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_MAX_YEAR = 2017

BEST_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.1,
    "subsample":     0.8,
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["log_pib_per_capita"] = np.log1p(df["pib_per_capita"])
    df = df.drop(columns=["pib_per_capita"])

    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=False)
    df = pd.concat([df, region_dummies], axis=1)

    drop_cols = ["iso3", "pais", "region", "esperanza_vida"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["esperanza_vida"]
    return X, y


def evaluar(nombre: str, y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {nombre:<28} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
    return {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4)}


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRENAMIENTO - Life Expectancy ML")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR / "dataset_final.csv")
    X, y = preparar_features(df)
    print(f"\nFeatures: {X.shape[1]}")

    print("\nEvaluacion:")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    evaluar("Split aleatorio 80/20", y_te, XGBRegressor(**BEST_PARAMS).fit(X_tr, y_tr).predict(X_te))

    es_train = X["anio"] <= TRAIN_MAX_YEAR
    model = XGBRegressor(**BEST_PARAMS).fit(X[es_train], y[es_train])
    metricas = evaluar(f"Split temporal (<= {TRAIN_MAX_YEAR})", y[~es_train], model.predict(X[~es_train]))

    salida = MODELS_DIR / "dashboard_model.pkl"
    joblib.dump({
        "model":          model,
        "feature_names":  list(X.columns),
        "train_max_year": TRAIN_MAX_YEAR,
        "metrics":        metricas,
    }, salida)
    print(f"\nModelo guardado en: {salida}")
    print("Listo.")
