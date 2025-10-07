# NLP Email Classification API

## Overview

This project implements a **semantic email classification system** exposed as a REST API using **FastAPI**.
The service classifies incoming emails into business categories (e.g. `facturacion`, `soporte`) and integrates business logic such as debt verification.

The workflow:

1. The API checks whether a given client has unpaid invoices (via MySQL).
2. If the client has no debts, the system classifies the email using a **LightGBM model** trained on **text embeddings**.
3. The result is returned through a FastAPI endpoint.

---

## Architecture

```
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── classifier.py           # Loads and executes the ML model
│   └── utils/                  # Preprocessing, helpers, etc.
│
├── scripts/
│   ├── train.py                # Model training pipeline
│   ├── eval.py                 # Model evaluation
│   ├── augment_data.py         # Synthetic data generation
│   └── create_split.py         # Train/test split
│
├── data/                       # CSV data (train/test, outputs)
├── model/                      # Saved model (.joblib)
├── Dockerfile
├── compose.yaml
└── requirements.txt
```

---

## API Definition

**Endpoint:** `/classify-email`
**Method:** `POST`
**Content-Type:** `application/json`

### Request body

```json
{
  "client_id": 3,
  "fecha_envio": "2022-05-11T06:26:11",
  "email_body": "¿Cada cuánto facturáis el gas? Llevo con vosotros desde el 10 de febrero y no me ha llegado ninguna factura."
}
```

### Successful response (client without debts)

```json
{
  "exito": true,
  "prediccion": "facturacion"
}
```

### Response when the client has debts

```json
{
  "exito": false,
  "razon": "El cliente tiene impagos"
}
```

---

## Core Components

### 1. FastAPI Application (`main.py`)

| Method | Route             | Description                       |
| ------ | ----------------- | --------------------------------- |
| GET    | `/health`         | Health check endpoint             |
| POST   | `/classify-email` | Checks debts and classifies email |

**Logic:**

* Receives a JSON payload with `client_id`, `fecha_envio`, and `email_body`.
* Queries MySQL for unpaid clients.
* If found → returns `{ "exito": false, "razon": "El cliente tiene impagos" }`.
* Otherwise → invokes the ML model and returns `{ "exito": true, "prediccion": "soporte" }`.

---

### 2. Classifier (`classifier.py`)

Bridges the API and the model stored on disk.

* Loads model via `joblib.load('/app/model/text_classifier.joblib')`
* Preprocesses input text (cleaning, embedding, PCA)
* Performs prediction and returns label (`"facturacion"`, `"soporte"`, etc.)

---

### 3. Model Training (`scripts/train.py`)

Builds and saves the classification model.

**Pipeline structure:**

```
Preprocessor → RandomOverSampler → LightGBMClassifier
```

**Main stages:**

1. **Data cleaning:** normalize labels, group rare classes, map to final targets (`facturacion`, `soporte`).
2. **Feature extraction:** SentenceTransformer embeddings.
3. **Class balancing:** `RandomOverSampler`.
4. **Hyperparameter tuning:** `GridSearchCV`.
5. **Cross-validation** for performance estimation.
6. **Model persistence:** stored as `/app/model/text_classifier.joblib`.

---

### 4. Evaluation (`scripts/eval.py`)

Evaluates the trained model on a validation dataset (`test_labels.csv`).

Generates:

* Classification report
* Confusion matrix
* CSV with misclassified examples

All outputs are saved to `/app/data/`.

---

### 5. Infrastructure (Docker)

`docker compose` orchestrates both the API and the database:

| Service | Description                                  |
| ------- | -------------------------------------------- |
| `db`    | MySQL 5.7 container with the `impagos` table |
| `app`   | FastAPI container with the trained model     |

**Environment variables:**

* `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME` for database access
* `HF_HOME`, `MODEL_PATH`, etc. for model and cache configuration

---

## Full Workflow

```bash
# 1. Build the image
docker compose build

# 2. Generate data and train the model
docker compose run --rm app python scripts/augment_data.py
docker compose run --rm app python scripts/create_split.py
docker compose run --rm app python scripts/train.py
docker compose run --rm app python scripts/eval.py

# 3. Launch the API
docker compose up -d

# 4. Test via Swagger UI
http://localhost:8000/docs
```

---

## Example Query

```bash
curl -X POST "http://localhost:8000/classify-email" \
  -H "Content-Type: application/json" \
  -d '{
        "client_id": 3,
        "fecha_envio": "2022-05-11T06:26:11",
        "email_body": "¿Cada cuánto facturáis el gas? Llevo con vosotros desde el 10 de febrero y no me ha llegado ninguna factura."
      }'
```

Response:

```json
{
  "exito": true,
  "prediccion": "facturacion"
}
```

---

## Technologies Used

* **FastAPI** — REST API framework
* **LightGBM** — gradient boosting classifier
* **Imbalanced-learn** — data balancing (`RandomOverSampler`)
* **SentenceTransformers** — text embeddings
* **Scikit-learn** — model evaluation and pipelines
* **SQLAlchemy** — database ORM
* **Docker Compose** — environment orchestration
* **MySQL 5.7** — client data storage

---

## Author

**Gerard** — Data Scientist in training
