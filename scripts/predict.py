import os
import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import utils.model_components 

def load_model(model_path: str):
    if not os.path.isfile(model_path):
        print(f"[ERROR] No se encuentra el modelo en: {model_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Cargando modelo desde {model_path} ...", flush=True)
    model = joblib.load(model_path)
    # Recuperamos las clases del clasificador
    try:
        classes = model.named_steps["clf"].classes_
    except Exception:
        # Fallback si cambia el nombre del paso
        classes = model.steps[-1][1].classes_
    return model, classes

def topk_from_proba(proba_row: np.ndarray, classes: np.ndarray, k: int = 3):
    idx = np.argsort(proba_row)[::-1][:k]
    return list(zip(classes[idx], proba_row[idx]))

def predict_text(model, classes, text: str, k: int = 3):
    df = pd.DataFrame({"email": [text]})
    pred = model.predict(df)[0]
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(df)[0]
        topk = topk_from_proba(proba, classes, k)
    else:
        proba, topk = None, None
    return pred, proba, topk

def predict_csv(model, classes, csv_path: str, out_path: str, k: int = 3, encoding: str = "utf-8"):
    if not os.path.isfile(csv_path):
        print(f"[ERROR] No existe el CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding=encoding)
    if "email" not in df.columns:
        print("[ERROR] El CSV debe tener una columna 'email'", file=sys.stderr)
        sys.exit(1)

    preds = model.predict(df[["email"]])
    df_out = df.copy()
    df_out["pred"] = preds

    # Probabilidades (si están disponibles)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(df[["email"]])
        # añadimos columnas proba_<clase>
        for i, cls in enumerate(classes):
            df_out[f"proba_{cls}"] = proba[:, i]
        # Top-k compactado (string "clase:prob; ...")
        topk_list = []
        for row in proba:
            tk = topk_from_proba(row, classes, k)
            topk_list.append("; ".join([f"{c}:{p:.3f}" for c, p in tk]))
        df_out[f"top{k}"] = topk_list

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    # Resumen por clase
    counts = df_out["pred"].value_counts().to_dict()
    return out_path, counts

def parse_args():
    p = argparse.ArgumentParser(description="Inferencia con text_classifier.joblib")
    p.add_argument("--model-path", default="/app/model/text_classifier.joblib",
                   help="Ruta al modelo .joblib")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Texto único a clasificar (string)")
    g.add_argument("--csv", help="Ruta CSV con columna 'email'")
    p.add_argument("--out", default="/app/data/predictions.csv",
                   help="Ruta de salida para predicciones de CSV")
    p.add_argument("--k", type=int, default=3, help="Top-k probabilidades a mostrar")
    p.add_argument("--encoding", default="utf-8", help="Encoding del CSV de entrada")
    return p.parse_args()

def main():
    # Recomendación para evitar sobre-paralelización
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    model, classes = load_model(args.model_path)

    if args.text is not None:
        pred, proba, topk = predict_text(model, classes, args.text, args.k)
        print("\n=== Predicción ===")
        print(f"label: {pred}")
        if topk is not None:
            print(f"\nTop-{args.k} probabilidades:")
            for c, p in topk:
                print(f"  - {c}: {p:.3f}")
    else:
        out_path, counts = predict_csv(model, classes, args.csv, args.out, args.k, args.encoding)
        print(f"\n[OK] Predicciones guardadas en: {out_path}")
        print("Conteo por clase predicha:", counts)

if __name__ == "__main__":
    main()
