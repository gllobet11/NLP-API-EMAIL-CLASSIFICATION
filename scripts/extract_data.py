import pandas as pd
from sqlalchemy import create_engine
DB_USER="root"; DB_PASS="root"; DB_HOST="db"; DB_NAME="atc"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4",
    pool_pre_ping=True,
)

# Query: emails de clientes que NO están en impagos
query = """
SELECT e.id, e.client_id, e.email
FROM emails e
LEFT JOIN impagos i ON e.client_id = i.client_id
WHERE i.client_id IS NULL;
"""

# Cargar en DataFrame
df = pd.read_sql(query, engine)

print("Número de correos válidos:", len(df))
print(df.head())
