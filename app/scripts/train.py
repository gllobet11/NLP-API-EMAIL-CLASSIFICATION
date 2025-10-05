import os
import logging
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sqlalchemy import create_engine

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from utils.model_components import Preprocessor  # <-- ahora importamos del módulo
from utils.model_components import KW_CONTRATOS, KW_FACTURACION, KW_SOPORTE
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_and_prepare_data() -> pd.DataFrame:
    """
    Carga los datos de entrenamiento directamente desde el archivo CSV pre-dividido.
    Ya no necesita conectarse a la base de datos.
    """
    logging.info("Cargando datos de entrenamiento desde /app/data/train_labels.csv...")
    
    # --- ÚNICO PASO: Cargar el CSV de entrenamiento ---
    # Este archivo ya contiene 'id', 'email' y 'categoria'
    df = pd.read_csv("/app/data/train_labels.csv") 
    
    # El resto del preprocesamiento se mantiene igual
    df["categoria"] = df["categoria"].astype(str).str.strip().str.lower().replace({"nan": None})
    df.dropna(subset=["categoria"], inplace=True)

    counts = df["categoria"].value_counts()
    raras = counts[counts < 2].index
    if not raras.empty:
        df.loc[df["categoria"].isin(raras), "categoria"] = "otros"

    class_mapping = {
        "gestion_cliente": "soporte", "incidencias": "soporte", "otros": "soporte",
        "acceso_plataforma": "soporte", "consumo_lecturas": "soporte", "info_comercial": "soporte",
    }
    df["categoria"] = df["categoria"].replace(class_mapping)
    logging.info(f"Distribución por clase: {df['categoria'].value_counts().to_dict()}")
    return df

def main():
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/model", exist_ok=True)

    df = load_and_prepare_data()
    X = df[['email']]
    y = df['categoria']

    preprocessor = Preprocessor()

    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('ros', RandomOverSampler(random_state=42)),
        ('clf', lgb.LGBMClassifier(random_state=42, n_jobs=1))
    ])

    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.1],
        'clf__num_leaves': [15, 20]
    }

    min_class = df['categoria'].value_counts().min()
    n_splits = max(2, min(5, int(min_class)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    logging.info("Iniciando búsqueda de hiperparámetros con LightGBM...")
    gs = GridSearchCV(model_pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=1, verbose=2, refit=True)
    gs.fit(X, y)

    best_est = gs.best_estimator_
    logging.info(f"[BEST] model=lgbm_hybrid params={gs.best_params_} | CV f1_macro={gs.best_score_:.3f}")

    logging.info("Evaluando modelo final (OOF) y generando outputs...")
    y_pred = np.empty_like(y.values, dtype=object)
    for i, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        logging.info(f"[CV] Entrenando fold {i}/{cv.get_n_splits()}...")
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        pipeline_clone = clone(best_est)
        pipeline_clone.fit(X_train, y_train)
        y_pred[te_idx] = pipeline_clone.predict(X_test)
        logging.info(f"[CV] Fold {i} completado.")

    try:
        labels_sorted = np.unique(y)
        report_text = classification_report(y, y_pred, digits=3)
        cm = confusion_matrix(y, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

        print("\n--- Reporte de Clasificación (Out-of-Fold) ---\n", flush=True)
        print(report_text, flush=True)
        print("\n--- Matriz de Confusión (Out-of-Fold) ---\n", flush=True)
        print(cm_df, flush=True)

        with open("/app/data/report.txt", "w", encoding="utf-8") as f:
            f.write("--- Reporte de Clasificación (Out-of-Fold) ---\n\n")
            f.write(report_text + "\n\n")
        cm_df.to_csv("/app/data/confusion_matrix.csv", index=True, encoding="utf-8")

        error_indices = np.where(y_pred != y.values)[0]
        errors_df = df.iloc[error_indices].copy()
        errors_df["predicted_categoria"] = y_pred[error_indices]
        errors_df[["id", "categoria", "predicted_categoria", "email"]].to_csv(
            "/app/data/misclassified_emails.csv", index=False, encoding="utf-8"
        )

        # Guarda comprimido para bajar tamaño (opcional)
        joblib.dump(best_est, "/app/model/text_classifier.joblib", compress=3)

    except Exception:
        with open("/app/data/train_error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        logging.exception("Fallo generando los outputs. Traza escrita en /app/data/train_error.txt")
        raise

if __name__ == "__main__":
    main()
