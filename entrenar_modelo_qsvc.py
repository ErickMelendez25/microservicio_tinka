import numpy as np
import pandas as pd
import os
import mysql.connector
from dotenv import load_dotenv
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import time

# 📥 1. Cargar variables de entorno
print("🔧 Cargando variables de entorno...")
load_dotenv()

# 🔌 2. Conectar a la base de datos y obtener los sorteos
print("🧠 Conectando a base de datos...")
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT", 3306))
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM sorteos ORDER BY fecha ASC")
sorteos = cursor.fetchall()
cursor.close()
conn.close()
print(f"📊 Registros obtenidos: {len(sorteos)}")

# 🧹 3. Preprocesamiento
df = pd.DataFrame(sorteos).dropna(subset=[f"bola{i}" for i in range(1, 7)])
X = df[[f"bola{i}" for i in range(1, 7)]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df[[f"bola{i}" for i in range(1, 7)]].mean(axis=1) // 10  # Clase según promedio

# 🔀 4. Seleccionar subconjunto para entrenamiento
print("🔀 Seleccionando 200 muestras para entrenamiento...")
X_train, _, y_train, _ = train_test_split(X_scaled, y, train_size=200, random_state=42)

# ⚛️ 5. Definir modelo cuántico con kernel moderno
print("⚛️ Inicializando QSVC con FidelityQuantumKernel...")
feature_map = ZZFeatureMap(feature_dimension=6, reps=1)
kernel = FidelityQuantumKernel(feature_map=feature_map)
model = QSVC(quantum_kernel=kernel)

# 🚀 6. Entrenar el modelo
print("🚀 Entrenando modelo cuántico...")
start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f"✅ Entrenamiento completado en {round(end - start, 2)} segundos.")

# 💾 7. Guardar modelo y scaler
print("💾 Guardando modelo y scaler...")
dump(model, "modelo_qsvc_tinka.joblib")
dump(scaler, "scaler_qsvc_tinka.joblib")
print("✅ Todo guardado correctamente.")
