import re, unicodedata, os
import pandas as pd

IN = "/app/data/labels_template.csv"
OUT = "/app/data/labels_template.csv"  # sobreescribimos

# --- limpieza idéntica a la exploración ---
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("\u00a0", " ")      # nbsp
    s = s.replace("\u00ad", "")       # soft hyphen
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", s)  # zero width & bidi
    s = re.sub(r"(\w)[\-\u00ad]\s+(\w)", r"\1\2", s)          # electro- nico -> electronico
    s = s.lower()
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- categorías finales (6) y prioridad de reglas ---
CATS = [
    "facturacion",
    "contratos",
    "acceso_plataforma",
    "gestion_cliente",
    "consumo_lecturas",
    "incidencias",
    "info_comercial",
]

RULES = {
    # facturas, cargos, importes, duplicados, avisos
    "facturacion": r"\bfactur|importe|cargo|cobro|duplicad[oa]|desglose|he recibido.*(factur|cargo)|enviar( me)? .*factur|factura mes|no me (llega[n]?|han enviado) .*factur",
    # contrato, duplicados, enviar contrato
    "contratos": r"\bcontrat|duplicado .*contrato|enviar( me)? .*contrato|copia .*contrato",
    # acceso/registro/descarga web/contraseña
    "acceso_plataforma": r"\bacceso|registr|login|contrasen|no puedo (acceder|entrar)|no puedo descargar .*factur|error .*servidor|desde la web|no puedo ver .*factur",
    # bajas, altas, cambios de datos/titularidad/cuenta/dirección/potencia
    "gestion_cliente": r"\bdar de baja|baja( del)?|alta( de)?|cambio .*titularidad|actualizar .*datos|cambiar .*cuenta|cuenta bancaria|cambiar .*direccion|cambiar .*potencia",
    # lecturas, históricos, curvas
    "consumo_lecturas": r"\blectur|historial de consumo|curvas de consumo",
    # incidencias técnicas, contador, cortes
    "incidencias": r"\bincidenc|suministro|contador|cortes programad|asistencia tecnica|tecnico",
    # tarifas, promos, ofertas, mejorar tarifa
    "info_comercial": r"\btarifa|promocion|oferta|mejorar .*tarifa|planes de energia",
}

def assign_category(text_clean: str) -> str:
    for cat in CATS:               # prioridad
        pat = RULES.get(cat)
        if pat and re.search(pat, text_clean):
            return cat
    return "otros"  # si ninguna regla aplica

if __name__ == "__main__":
    df = pd.read_csv(IN)
    df["text_clean"] = df["email"].apply(clean_text)
    df["categoria"] = df["text_clean"].apply(assign_category)

    # fusiona categorías raras en las 6 finales
    df.loc[df["categoria"]=="otros", "categoria"] = "facturacion"  # fallback pragmático

    vc = df["categoria"].value_counts()
    print("Distribución inicial:\n", vc)

    # guardamos encima
    df[["id","client_id","email","categoria"]].to_csv(OUT, index=False)
    print(f"Guardado: {OUT}")
