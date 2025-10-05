# /app/model_components.py
import os
import re
import unicodedata
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer


# Config
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL = "/app/model_cache/paraphrase-multilingual-MiniLM-L12-v2"

# Diccionarios de keywords (se usan al construir el pipeline)
KW_CONTRATOS = {
    "contrato": r"\bcontrato\b|condiciones generales|firmado|firmar|firmante|copia del contrato",
    "titularidad": r"cambio .*titular|titular del contrato|cambio de nombre",
    "domicilio": r"cambi(ar|o) .*direccion|domicilio",
    "contratar": r"\bcontratar\b|nuevo servicio|nueva alta|dar de alta",
    "potencia": r"cambiar la potencia|potencia contratada"
}
KW_FACTURACION = {
    "factura": r"\bfactura(s)?\b|recibo|facturacion",
    "pago": r"\bpago\b|pagar|cargo duplicado",
    "cuenta_bancaria": r"cuenta bancaria|iban|domiciliacion",
    "descargar": r"descargar|no puedo ver|visualizar",
    "consumo": r"consumo|lectura|historial",
    "documentacion": r"duplicado|enviarmelas|copia de la factura"
}
KW_SOPORTE = {
    "baja_servicio": r"\bdar(me)? de baja\b|baja( del)? servicio",
    "alta_servicio": r"alta .*punto de suministro|alta de suministro",
    "acceso": r"\bacceder\b|entrar|contrasena|restablecer|usuario|regist|olvidado|no puedo acceder",
    "datos_personales": r"actualizar mis datos|datos personales|informacion de contacto|numero de cliente",
    "incidencia": r"incidenc|averia|no funciona|asistencia tecnica|tecnico|reportar|contador",
    "corte_suministro": r"corte(s)? (de luz|programado)|suministro electrico",
    "informacion": r"informacion|promociones|duda|aclarar|saber si hay",
    "consulta": r"consulta|pregunta|saber"
}

def select_text_column(df: pd.DataFrame) -> pd.Series:
    """Función picklable usada en FunctionTransformer para seleccionar la columna 'text'."""
    return df['text']

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy['text'] = X_copy['email'].apply(self._clean_text_series)
        return X_copy
    def _clean_text_series(self, s: str) -> str:
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

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = LOCAL if os.path.isdir(LOCAL) else MODEL_ID
        self.model_name = model_name
        logging.info(f"Use SentenceTransformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        logging.info(f"Generando embeddings con el modelo '{self.model_name}'...")
        return self.model.encode(X['text'].tolist(), show_progress_bar=True)


class KeywordFlags(BaseEstimator, TransformerMixin):
    def __init__(self, patterns):
        self.patterns = patterns  # dict o lista de (name, pattern)

    def fit(self, X, y=None):
        items = list(self.patterns.items()) if isinstance(self.patterns, dict) else list(self.patterns)
        self._compiled = [(name, re.compile(pat)) for name, pat in items]
        return self

    def transform(self, X):
        rows = []
        for txt in X:
            t = (txt or "").lower()
            rows.append([1.0 if rgx.search(t) else 0.0 for _, rgx in self._compiled])
        return csr_matrix(np.asarray(rows, dtype=np.float32))


def build_feature_pipeline(kw_contratos, kw_facturacion, kw_soporte) -> Pipeline:
    keyword_flags = FeatureUnion([
        # Ahora usa los argumentos que le pasaste
        ("flags_contratos",   KeywordFlags(kw_contratos)),
        ("flags_facturacion", KeywordFlags(kw_facturacion)),
        ("flags_soporte",     KeywordFlags(kw_soporte))
    ])
    embedding_pipeline = Pipeline([
        ('embeddings', EmbeddingTransformer()),
        ('pca', PCA(n_components=0.95, random_state=42))
    ])
    feature_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('features', FeatureUnion([
            ('embedding_features', embedding_pipeline),
            ('keyword_features', Pipeline([
                ('selector', FunctionTransformer(select_text_column, validate=False)),
                ('flags',    keyword_flags)
            ]))
        ]))
    ])
    return feature_pipeline

class Preprocessor(BaseEstimator, TransformerMixin):
    """Wrapper para usar un sklearn.Pipeline dentro de imblearn.Pipeline."""

    def __init__(self):
        # Llama a la función build_feature_pipeline pasando las keywords
        self.pipe = build_feature_pipeline(
            kw_contratos=KW_CONTRATOS,
            kw_facturacion=KW_FACTURACION,
            kw_soporte=KW_SOPORTE
        )
    def fit(self, X, y=None):
        self.pipe.fit(X, y)
        return self
    def transform(self, X):
        return self.pipe.transform(X)

