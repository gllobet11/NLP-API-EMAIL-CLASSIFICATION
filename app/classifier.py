
import os
import re
import sys
import types
import unicodedata
import joblib
import pandas as pd
import logging


try:
    import utils.model_components as umc
    # Compatibilidad con artefactos antiguos
    sys.modules.setdefault("utils.model_components", umc)
    sys.modules.setdefault("model_components", umc)
except Exception as e:
    logging.error("No se pudo importar utils.model_components: %s", e)

MODEL_PATH = "/app/model/text_classifier.joblib"
_modelo_principal = None  # Cache interno del modelo


def get_model():
    """Carga el modelo en la primera llamada (lazy loading) y lo deja en memoria."""
    global _modelo_principal
    if _modelo_principal is None:
        if not os.path.exists(MODEL_PATH):
            logging.warning(f"⚠️ Modelo no encontrado en {MODEL_PATH}")
            raise FileNotFoundError(MODEL_PATH)
        _modelo_principal = joblib.load(MODEL_PATH)
        logging.info("✅ Modelo principal cargado en classifier.py")
    return _modelo_principal


def clean_text(s: str) -> str:
    """Debe ser idéntica a la función de limpieza usada en el entrenamiento."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("\u00a0", " ").replace("\u00ad", "")
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", s)
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---- Reglas de negocio para la predicción en dos etapas ----
FACTURACION_IMPORTE = [
    r"\bimporte(s)?\b", r"\bcargo(s)? duplicad", r"\bcobro(s)? duplicad",
    r"\b(reclamaci[oó]n|reclamar)\b.*factur", r"\berror(es)? en la factura\b",
    r"\bfactura (err[oó]nea|mal|incorrecta)\b", r"\babono\b", r"\bdevoluci[oó]n\b",
    r"\bregulariz(ar|aci[oó]n)\b", r"\brevis(ar|i[oó]n)\b.*factur", r"\bimporte muy (alto|elevado|excesivo)\b",
]
FACTURACION_DOCS = [
    r"\b(enviar|mandar|remitir)\b.*factur", r"\b(duplicado|copia)\b.*factur",
    r"\bdescargar\b.*factur(?!.*\bno puedo\b)",
]
SOPORTE_ACCESO = [
    r"\bno puedo\b.*\b(acceder|entrar|iniciar sesi[oó]n|log(in|uear)|descargar|ver|visualizar)\b",
    r"\bcontrase(ñ|n)a\b", r"\busuario\b", r"\berror( de)? acceso\b",
    r"\b(portal|web|app|plataforma)\b",
]

# Compila las expresiones regulares una sola vez para mayor eficiencia
RX_FACT_IMPORTE = [re.compile(p, re.IGNORECASE) for p in FACTURACION_IMPORTE]
RX_FACT_DOCS    = [re.compile(p, re.IGNORECASE) for p in FACTURACION_DOCS]
RX_SUP_ACCESO   = [re.compile(p, re.IGNORECASE) for p in SOPORTE_ACCESO]


def _score_rules(texto_limpio: str) -> dict:
    """Devuelve una puntuación por categoría según los patrones encontrados."""
    scores = {"facturacion": 0, "soporte": 0}

    for rx in RX_SUP_ACCESO:
        if rx.search(texto_limpio):
            scores["soporte"] += 2
    for rx in RX_FACT_IMPORTE:
        if rx.search(texto_limpio):
            scores["facturacion"] += 2
    for rx in RX_FACT_DOCS:
        if rx.search(texto_limpio):
            scores["facturacion"] += 1

    # Regla especial: 'no puedo' + 'factura' suele indicar un problema de acceso (soporte)
    if re.search(r"\bno puedo\b", texto_limpio) and re.search(r"\bfactur", texto_limpio):
        scores["soporte"] += 1

    return scores


def predecir_categoria(email_texto: str) -> str:
    """
    Clasifica un email usando una estrategia híbrida: reglas de negocio + modelo de ML.
    """
    try:
        model = get_model()
    except FileNotFoundError:
        return "error_modelo_no_cargado"

    texto_limpio = clean_text(email_texto)
    scores = _score_rules(texto_limpio)

    # 1. Si las reglas dan una ventaja clara, las usamos
    if scores["soporte"] >= scores["facturacion"] + 2:  # Umbral de 2 para más seguridad
        return "soporte"
    if scores["facturacion"] >= scores["soporte"] + 2:
        return "facturacion"

    # 2. Si hay ambigüedad, decidimos con el modelo de ML
    input_df = pd.DataFrame([{"email": email_texto}])
    pred = model.predict(input_df)[0]

    return pred