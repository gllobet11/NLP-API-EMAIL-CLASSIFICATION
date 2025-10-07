import pandas as pd
from sklearn.model_selection import train_test_split

# Carga el archivo de etiquetas aumentado
INPUT_FILE = "/app/data/labels_augmented.csv"
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Cargado con éxito el dataset aumentado de {INPUT_FILE}")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} no encontrado. Asegúrate de ejecutar augment_data.py primero.")
    exit()


df["categoria"] = df["categoria"].astype(str).str.strip().str.lower().replace({"nan": None})
df.dropna(subset=["categoria"], inplace=True)

class_mapping = {
    "gestion_cliente": "soporte", "incidencias": "soporte", "otros": "soporte",
    "acceso_plataforma": "soporte", "consumo_lecturas": "soporte", "info_comercial": "soporte",
}
df["categoria"] = df["categoria"].replace(class_mapping)

print("Distribución de clases final antes de dividir:")
print(df['categoria'].value_counts())

# Realiza la división estratificada
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['categoria']
)

# Guarda los nuevos archivos
train_df.to_csv("/app/data/train_labels.csv", index=False)
test_df.to_csv("/app/data/test_labels.csv", index=False)

print("\nDivisión completada:")
print(f" -> {len(train_df)} ejemplos guardados en /app/data/train_labels.csv")
print(f" -> {len(test_df)} ejemplos guardados en /app/data/test_labels.csv")