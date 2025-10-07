import os
import logging
import warnings
import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from utils.model_components import Preprocessor

# Config básica de logs y warnings
warnings.filterwarnings("ignore")  # en dev: podrías filtrar por categoría concreta
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TRAIN_CSV = "/app/data/train_labels.csv"
MODEL_PATH = "/app/model/text_classifier.joblib"
OUT_DIR = "/app/data"
RANDOM_STATE = 42


def load_and_prepare_data() -> pd.DataFrame:
    """
    Carga el CSV de entrenamiento y normaliza etiquetas:
    - baja a minúsculas, quita 'nan'
    - colapsa clases raras en 'otros'
    - mapea varias categorías al esquema final binario: {facturacion, soporte}
    """
    logging.info(f"Cargando datos de entrenamiento desde {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)

    # Normaliza etiquetas
    df["categoria"] = (
        df["categoria"].astype(str).str.strip().str.lower().replace({"nan": None})
    )
    df.dropna(subset=["categoria"], inplace=True)

    # Fusiona clases con muy pocos ejemplos en 'otros'
    counts = df["categoria"].value_counts()
    raras = counts[counts < 2].index
    if len(raras) > 0:
        df.loc[df["categoria"].isin(raras), "categoria"] = "otros"

    # Mapea al esquema final (binario)
    mapping = {
        "gestion_cliente": "soporte",
        "incidencias": "soporte",
        "otros": "soporte",
        "acceso_plataforma": "soporte",
        "consumo_lecturas": "soporte",
        "info_comercial": "soporte",
        # 'facturacion' queda como está
    }
    df["categoria"] = df["categoria"].replace(mapping)

    logging.info(f"Distribución por clase: {df['categoria'].value_counts().to_dict()}")
    return df


def build_pipeline() -> ImbPipeline:
    """
    Construye el pipeline de entrenamiento:
    - Preprocessor: limpieza + embeddings (ST) + PCA + flags
    - RandomOverSampler: balanceo de clases en fit
    - LightGBM: clasificador final
    """
    preprocessor = Preprocessor()
    pipe = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
            ("clf", lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    return pipe


def make_cv(y: pd.Series) -> StratifiedKFold:
    """
    Devuelve un CV estratificado con n_splits seguro según la clase minoritaria.
    """
    min_class = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class))  # evita folds inviables
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def grid_and_fit(
    X: pd.DataFrame, y: pd.Series, base_pipeline: ImbPipeline, cv: StratifiedKFold
) -> GridSearchCV:
    """
    Ejecuta GridSearchCV sobre LightGBM (solo unos pocos params para ir rápido).
    """
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__learning_rate": [0.1],
        "clf__num_leaves": [15, 20],
    }
    logging.info("Iniciando búsqueda de hiperparámetros con LightGBM...")
    gs = GridSearchCV(
        base_pipeline,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,      # en contenedores pequeños mejor 1
        verbose=2,
        refit=True,    # ajusta con los mejores params al final
    )
    gs.fit(X, y)
    logging.info(
        f"[BEST] model=lgbm_hybrid params={gs.best_params_} | CV f1_macro={gs.best_score_:.3f}"
    )
    return gs


def oof_evaluation(
    X: pd.DataFrame, y: pd.Series, best_estimator: ImbPipeline, cv: StratifiedKFold
) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Evalúa out-of-fold: reentrena en cada fold con el mejor pipeline y
    predice sobre el fold de test para estimar rendimiento honesto.
    """
    y_pred = np.empty_like(y.values, dtype=object)
    for i, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        logging.info(f"[CV] Entrenando fold {i}/{cv.get_n_splits()}...")
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]

        # clone() para no contaminar el objeto base entre folds
        pipe = clone(best_estimator)
        pipe.fit(X_tr, y_tr)
        y_pred[te_idx] = pipe.predict(X_te)

    labels_sorted = np.unique(y)
    report = classification_report(y, y_pred, digits=3)
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    return y_pred, report, cm_df


def save_outputs(report: str, cm_df: pd.DataFrame, df: pd.DataFrame, y_pred: np.ndarray) -> None:
    """
    Persiste reportes y misclasificados para inspección rápida.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Reporte OOF
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("--- Reporte de Clasificación (Out-of-Fold) ---\n\n")
        f.write(report + "\n\n")

    # Matriz de confusión
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=True, encoding="utf-8")

    # Misclasificados
    err_idx = np.where(y_pred != df["categoria"].values)[0]
    errors_df = df.iloc[err_idx].copy()
    errors_df["predicted_categoria"] = y_pred[err_idx]
    errors_df[["id", "categoria", "predicted_categoria", "email"]].to_csv(
        os.path.join(OUT_DIR, "misclassified_emails.csv"), index=False, encoding="utf-8"
    )


def main():
    os.makedirs("/app/model", exist_ok=True)

    df = load_and_prepare_data()
    X, y = df[["email"]], df["categoria"]

    base_pipeline = build_pipeline()
    cv = make_cv(y)

    try:
        gs = grid_and_fit(X, y, base_pipeline, cv)
        best_est = gs.best_estimator_

        logging.info("Evaluando modelo final (OOF) y generando outputs...")
        y_pred, report, cm_df = oof_evaluation(X, y, best_est, cv)

        print("\n--- Reporte de Clasificación (Out-of-Fold) ---\n", report, flush=True)
        print("\n--- Matriz de Confusión (Out-of-Fold) ---\n", cm_df, flush=True)

        save_outputs(report, cm_df, df, y_pred)

        # Guarda el pipeline completo (requiere imbalanced-learn en inferencia)
        joblib.dump(best_est, MODEL_PATH, compress=3)
        logging.info(f"Modelo guardado en {MODEL_PATH}")

    except Exception:
        # Log a fichero para inspección post-mortem
        with open(os.path.join(OUT_DIR, "train_error.txt"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        logging.exception("Fallo generando los outputs. Traza escrita en /app/data/train_error.txt")
        raise


if __name__ == "__main__":
    main()
