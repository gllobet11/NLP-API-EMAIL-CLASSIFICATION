import os, re, unicodedata
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer

# --------- Config DB ----------
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "atc")
ENGINE = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4",
    pool_pre_ping=True,
)

QUERY = """
SELECT e.id, e.client_id, e.email
FROM emails e
LEFT JOIN impagos i ON e.client_id = i.client_id
WHERE i.client_id IS NULL;
"""

# --------- Utilidades texto ----------

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def clean_text(s: str) -> str:
    if not s:
        return ""
    # normaliza saltos/espacios raros
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("\u00a0", " ")          # NBSP
    s = s.replace("\u00ad", "")           # soft hyphen
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", s)  # zero-width & bidi

    # repara palabras cortadas por guion: "electro- nico" -> "electronico"
    s = re.sub(r"(\w)[\-\u00ad]\s+(\w)", r"\1\2", s)

    s = s.lower()
    s = strip_accents(s)

    # whitelist: letras/dígitos/espacio
    s = re.sub(r"[^a-z0-9 ]", " ", s)

    # compacta espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Stopwords españolas básicas (sin libs extra)
SPANISH_STOP = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
    "una","su","al","lo","como","mas","pero","sus","le","ya","o","fue","ha","si","porque",
    "muy","sin","sobre","tambien","me","hasta","hay","donde","quien","desde","todo","nos",
    "durante","todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto",
    "mi","antes","algunos","que","unos","yo","otro","otras","otra","el","tanto","esa","estos",
    "mucho","quienes","nada","muchos","cual","poco","ella","estar","estas","algunas","algo",
    "nosotros","mi","mis","tu","te","ti","tu","tus","ellas","nosotras","vosotros","vosotras",
    "os","mio","mia","mios","mias","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos","suyas",
    "nuestro","nuestra","nuestros","nuestras","vuestro","vuestra","vuestros","vuestras","esos",
    "esas","estoy","estas","esta","estamos","estais","estan","este","estes","estan","ser","es",
    "soy","eres","somos","sois","son","tengo","tienes","tiene","tenemos","teneis","tienen",
    "buenos","dias","buenas","gracias","hola"
}

# --------- Palabras clave por “tema” (heurística) ----------
TOPICS = {
   "acceso/plataforma": r"\b(acceso|registr|login|contrasena|no puedo (?:acceder|entrar)|error del servidor|web)\b",
    "facturacion": r"\bfactur|importe|cargo|cobro|desglose|duplicad[oa]\b",
    "acceso/plataforma": r"\bacceso|registr|login|contrasena|no puedo (acceder|entrar)|error del servidor|web\b",
    "contratos":         r"\bcontrat|duplicado de mi contrato|enviar.*contrato\b",
    "baja/alta":         r"\bbaja( del)?|dar de baja|alta( de)?\b",
    "datos/contacto":    r"\bactualizar (mis )?datos|cambiar (cuenta|direccion)|informacion de contacto\b",
    "consumo/lecturas":  r"\blectur|historial de consumo|curvas de consumo\b",
    "tarifas/ofertas":   r"\btarifa|promocion|oferta|mejorar mi tarifa\b",
    "incidencias":       r"\bincidenc|suministro|contador|cortes programados\b",
    "estado/pagos":      r"\baviso de impago|pago recibido|regularizar\b",
}


def topic_counts(text: str):
    out = {}
    for k, pattern in TOPICS.items():
        out[k] = 1 if re.search(pattern, text) else 0
    return out

# --------- Main ----------
if __name__ == "__main__":
    df = pd.read_sql(QUERY, ENGINE)
    df["text_clean"] = df["email"].apply(clean_text)

    # Conteos por tema heurístico
    topic_df = df["text_clean"].apply(topic_counts).apply(pd.Series)
    counts = topic_df.sum().sort_values(ascending=False)
    print("\n== Conteos por tema (heurístico) ==")
    print(counts.to_string())

    # Top unigrams
    vec1 = CountVectorizer(stop_words=list(SPANISH_STOP), ngram_range=(1,1), min_df=2, max_df=0.9)
    X1 = vec1.fit_transform(df["text_clean"])
    vocab1 = pd.Series(X1.toarray().sum(axis=0), index=vec1.get_feature_names_out()).sort_values(ascending=False)
    print("\n== Top palabras (unigramas) ==")
    print(vocab1.head(30).to_string())

    # Top bigramas
    vec2 = CountVectorizer(stop_words=list(SPANISH_STOP), ngram_range=(2,2), min_df=2, max_df=0.9)
    X2 = vec2.fit_transform(df["text_clean"])
    vocab2 = pd.Series(X2.toarray().sum(axis=0), index=vec2.get_feature_names_out()).sort_values(ascending=False)
    print("\n== Top bigramas ==")
    print(vocab2.head(30).to_string())

    # Muestras por tema (3 ejemplos cada uno)
    print("\n== Ejemplos por tema ==")
    for k, pattern in TOPICS.items():
        mask = df["text_clean"].str.contains(pattern,regex=True,na=False)
        examples = df.loc[mask, ["id", "text_clean"]].head(3)
        if not examples.empty:
            print(f"\n--- {k} ---")
            print(examples.to_string(index=False))

    # Guardados útiles para etiquetar / modelar
    os.makedirs("/app/data", exist_ok=True)
    df[["id","client_id","email"]].to_csv("/app/data/emails_validos.csv", index=False)
    counts.to_csv("/app/data/topic_counts_heuristic.csv", header=["count"])
    vocab1.head(200).to_csv("/app/data/top_unigrams.csv", header=["tf"])
    vocab2.head(200).to_csv("/app/data/top_bigrams.csv", header=["tf"])
    print("\nArchivos guardados en /app/data: emails_validos.csv, topic_counts_heuristic.csv, top_unigrams.csv, top_bigrams.csv")
