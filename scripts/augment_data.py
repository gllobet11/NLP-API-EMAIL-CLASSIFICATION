# /app/augment_data.py

import pandas as pd
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Carga los pipelines de traducción de Hugging Face
logging.info("Cargando modelos de traducción (puede tardar la primera vez)...")
translator_es_a_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translator_en_a_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
logging.info("Modelos cargados.")

def back_translate(text: str) -> str:
    """Aplica back-translation a un texto."""
    if not isinstance(text, str) or not text.strip():
        return text
    
    try:
        translated_en = translator_es_a_en(text, max_length=512)[0]['translation_text']
        back_translated_es = translator_en_a_es(translated_en, max_length=512)[0]['translation_text']
        return back_translated_es
    except Exception as e:
        logging.warning(f"No se pudo traducir el texto: '{text[:50]}...'. Error: {e}")
        return text

# Carga tus datos originales
try:
    df = pd.read_csv("/app/data/labels_template.csv")
    logging.info(f"Leídos {len(df)} ejemplos originales.")
except FileNotFoundError:
    logging.error("Error: /app/data/labels_template.csv no encontrado.")
    exit()

# Aplica la aumentación
logging.info("Iniciando proceso de back-translation...")
df['email_aumentado'] = df['email'].apply(back_translate)

# Crea el nuevo DataFrame aumentado
augmented_df = df[['id', 'email_aumentado', 'categoria']].copy()
augmented_df.rename(columns={'email_aumentado': 'email'}, inplace=True)

# Opcional: añade un prefijo al ID para evitar duplicados
augmented_df['id'] = augmented_df['id'].apply(lambda x: f"aug_{x}")

# Combina los datos originales y los aumentados
df_original = df[['id', 'email', 'categoria']]
df_final_aumentado = pd.concat([df_original, augmented_df], ignore_index=True)

# Guarda el nuevo dataset completo
output_path = "/app/data/labels_augmented.csv"
df_final_aumentado.to_csv(output_path, index=False)

logging.info(f"¡Proceso completado! Se ha creado un nuevo dataset con {len(df_final_aumentado)} ejemplos en: {output_path}")

##posible implementacion futura:
# import random

# SYNONYMS = {
#     "factura": ["recibo", "cobro"],
#     "contrato": ["acuerdo", "póliza"],
#     "problema": ["incidencia", "error", "fallo"],
#     "cambiar": ["modificar", "actualizar"],
# }

# def replace_synonyms(text: str) -> str:
#     words = text.split()
#     for i, word in enumerate(words):
#         if word in SYNONYMS and random.random() < 0.5: # 50% de probabilidad de reemplazar
#             words[i] = random.choice(SYNONYMS[word])
#     return " ".join(words)