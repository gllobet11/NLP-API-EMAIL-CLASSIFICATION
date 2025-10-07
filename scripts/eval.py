import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from app import classifier 
import utils.model_components as umc
import sys
sys.modules.setdefault("model_components", umc)  # alias si tu .joblib lo necesita

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUT_DIR = "/app/data"
MODEL_OK_MSG = "Modelo cargado y listo."
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "atc")

def load_gold() -> pd.DataFrame:
    """
    Carga los datos de evaluación directamente desde el archivo CSV de test.
    Ya no necesita conectarse a la base de datos.
    """
    logging.info("Cargando datos de test desde /app/data/test_labels.csv...")
    
    # ---  Cargar el CSV de test ---
    df = pd.read_csv("/app/data/test_labels.csv")
    
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

    return df[["id", "email", "categoria"]].reset_index(drop=True)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp_path = f"{OUT_DIR}/eval_report_{ts}.txt"
    cm_path = f"{OUT_DIR}/eval_confusion_matrix_{ts}.csv"
    err_path = f"{OUT_DIR}/eval_misclassified_{ts}.csv"

    logging.info("Cargando dataset etiquetado...")
    df = load_gold()
    X = df[["email"]]
    y_true = df["categoria"].values

    logging.info("Cargando modelo...")
    model = classifier.get_model()
    logging.info(MODEL_OK_MSG)

    logging.info("Inferencia sobre el conjunto etiquetado...")
    y_pred = model.predict(X)

    from sklearn.metrics import classification_report, confusion_matrix
    labels_sorted = np.unique(y_true)
    report_text = classification_report(y_true, y_pred, labels=labels_sorted, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    mis_idx = np.where(y_pred != y_true)[0]
    mis = df.iloc[mis_idx].copy()
    mis["predicted_categoria"] = y_pred[mis_idx]

    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(X)
        classes = model.named_steps["clf"].classes_
        cls2idx = {c: i for i, c in enumerate(classes)}
        mis["proba_pred"] = [proba[i, cls2idx[p]] for i, p in zip(mis_idx, mis["predicted_categoria"])]
        mis["proba_true"] = [proba[i, cls2idx.get(t, -1)] if t in cls2idx else np.nan
                             for i, t in zip(mis_idx, mis["categoria"])]
    else:
        mis["proba_pred"] = np.nan
        mis["proba_true"] = np.nan

    logging.info("Guardando resultados en /app/data ...")
    with open(rp_path, "w", encoding="utf-8") as f:
        f.write("--- Reporte de Clasificación (Eval) ---\n\n")
        f.write(report_text + "\n")

    cm_df.to_csv(cm_path, index=True, encoding="utf-8")
    mis[["id", "categoria", "predicted_categoria", "proba_pred", "proba_true", "email"]].to_csv(
        err_path, index=False, encoding="utf-8"
    )

    print("\n=== Reporte (Eval) ===\n")
    print(report_text, flush=True)
    print("\n=== Matriz de Confusión (Eval) ===\n")
    print(cm_df, flush=True)
    print(f"\n-> Misclasificados: {len(mis)} guardados en {err_path}")
    print(f"-> Reporte: {rp_path}")
    print(f"-> Confusion matrix: {cm_path}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
