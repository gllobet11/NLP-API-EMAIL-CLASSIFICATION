import os, pandas as pd
from sqlalchemy import create_engine

DB_USER=os.getenv("DB_USER","root")
DB_PASS=os.getenv("DB_PASS","root")
DB_HOST=os.getenv("DB_HOST","db")
DB_NAME=os.getenv("DB_NAME","atc")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4", pool_pre_ping=True)

df = pd.read_sql("""
SELECT e.id, e.client_id, e.email
FROM emails e
LEFT JOIN impagos i ON e.client_id = i.client_id
WHERE i.client_id IS NULL;
""", engine)

df["categoria"] = ""  # para que la completes a mano
df.to_csv("/app/data/labels_template.csv", index=False)
print("-> /app/data/labels_template.csv listo")
