import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mysql.connector

# Qiskit
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ======================
# Cargar .env
# ======================
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("DB_NAME")

print("\n=== DEBUG VARIABLES .ENV ===")
print(f"DB_HOST: {DB_HOST}")
print(f"DB_USER: {DB_USER}")
print(f"DB_PASSWORD: (oculto)")
print(f"DB_PORT: {DB_PORT}")
print(f"DB_NAME: {DB_NAME}")
print("============================")

# ======================
# Conectar y leer DB
# ======================
print("🔎 Leyendo sorteos desde DB (últimos 200 registros)...")

conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT,
    database=DB_NAME
)

cursor = conn.cursor()

# Seleccionamos bolas principales
cursor.execute("""
    SELECT bola1, bola2, bola3, bola4, bola5, bola6
    FROM sorteos
    ORDER BY fecha DESC
    LIMIT 200
""")
rows = cursor.fetchall()
conn.close()

if not rows:
    raise ValueError("⚠️ No hay registros en la tabla 'sorteos'.")

# ======================
# Preparar features
# ======================
X = np.array(rows)

# Etiquetas dummy (ejemplo: la primera bola como target)
y = X[:, 0]

print("🔢 Preparando X, y...")
print("📊 Calculando estadísticas de frecuencia...")
unique, counts = np.unique(X, return_counts=True)
freq = dict(zip(unique, counts))
print("🎯 Bolas más frecuentes:", sorted(freq, key=freq.get, reverse=True)[:10])

# ======================
# Escalado
# ======================
print("⚖️ Escalando X con StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_qsvc_tinka.joblib")
print("✅ Scaler guardado: scaler_qsvc_tinka.joblib")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======================
# Quantum Kernel SVM
# ======================
print(f"🎛️ Entrenando modelo cuántico (kernel) con {len(X_train)} registros...")

algorithm_globals.random_seed = 42
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)

# Aquí solo feature_map, nada de sampler
fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Entrenar SVM
svc = SVC(kernel=fidelity_kernel.evaluate)
svc.fit(X_train, y_train)

# Evaluar
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📈 Accuracy del modelo cuántico: {acc:.2f}")

# Guardar modelo
joblib.dump(svc, "qsvc_tinka_model.joblib")
print("💾 Modelo cuántico guardado: qsvc_tinka_model.joblib")
