import os
import logging
from typing import Union
from datetime import datetime  

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text


from .classifier import predecir_categoria

import utils.model_components as umc
import sys
sys.modules.setdefault("model_components", umc)

app = FastAPI(
    title="API de Clasificación de Emails",
    description="Clasifica emails comprobando primero si el cliente tiene impagos."
)

# --- Conexión a la Base de Datos ---
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "atc")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --- Modelos Pydantic ---
class ClassifyRequest(BaseModel):
    client_id: int
    fecha_envio: datetime  # cambiamos string a datetime
    email_body: str

class SuccessResponse(BaseModel):
    exito: bool = True
    prediccion: str

class FailureResponse(BaseModel):
    exito: bool = False
    razon: str = "El cliente tiene impagos"

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post(
    "/classify-email",
    response_model=Union[SuccessResponse, FailureResponse],
    summary="Clasifica un email tras verificar impagos"
)

def classify_email(item: ClassifyRequest):
    """
    Este endpoint clasifica el `email_body` de un cliente.
    
    - Primero, comprueba si el `client_id` está en la tabla de impagos.
    - Si tiene impagos, devuelve `exito: false`.
    - Si no tiene impagos, usa el modelo de ML para predecir la categoría y devuelve `exito: true`.
    """
    try:
        # Lógica para comprobar impagos
        logging.info(f"Comprobando impagos para client_id: {item.client_id}")
        with engine.connect() as connection:
            query = text("SELECT client_id FROM impagos WHERE client_id = :client_id LIMIT 1")
            result = connection.execute(query, {"client_id": item.client_id}).fetchone()

        if result:
            # Si el cliente tiene impagos, devuelve la respuesta de fallo
            logging.warning(f"Cliente {item.client_id} encontrado en la tabla de impagos.")
            return FailureResponse()
        
        # Si no hay impagos, se llama al clasificador
        logging.info(f"Cliente {item.client_id} sin impagos. Clasificando email...")
        categoria = predecir_categoria(item.email_body)
        
        return SuccessResponse(prediccion=categoria)

    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")