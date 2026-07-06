"""
Entrena el modelo final XGBoost con los hiperparametros optimos,
evalua en test set y guarda el modelo en models/best_model.pkl

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

BEST_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.05,
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
    print(f"  {nombre:<20} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
    return {"modelo": nombre, "rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRENAMIENTO - Life Expectancy ML")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR / "dataset_final.csv")
    X, y = preparar_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    print(f"Features: {X_train.shape[1]}")

    print("\nEntrenando modelo XGBoost con parametros optimos...")
    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    print("\nEvaluacion en test set:")
    evaluar("XGBoost", y_test, model.predict(X_test))

    salida = MODELS_DIR / "best_model.pkl"
    joblib.dump({"model": model, "feature_names": list(X_train.columns)}, salida)
    print(f"\nModelo guardado en: {salida}")
    print("Listo.")
