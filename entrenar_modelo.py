import numpy as np
import pandas as pd
import os
import mysql.connector
from dotenv import load_dotenv
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import Aer
from sklearn.model_selection import train_test_split
from joblib import dump
import time

# 📥 Cargar variables .env
print("🔧 Cargando variables de entorno...")
load_dotenv()

# Función para codificar bolas en vector one-hot
def quantum_encode(row):
    vec = np.zeros(50)
    for b in [f"bola{i}" for i in range(1, 7)]:
        vec[int(row[b]) - 1] = 1
    return vec

# 🔌 Conexión a base de datos
print("🧠 Conectando a base de datos...")
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT", 3306))
)
cursor = conn.cursor(dictionary=True)

print("📥 Obteniendo datos de la tabla 'sorteos'...")
cursor.execute("SELECT * FROM sorteos ORDER BY fecha ASC")
sorteos = cursor.fetchall()
cursor.close()
conn.close()
print(f"📊 Total de registros obtenidos: {len(sorteos)}")

# 🔢 Preprocesamiento
print("🧹 Limpiando y transformando datos...")
df = pd.DataFrame(sorteos).dropna(subset=[f"bola{i}" for i in range(1, 7)])
X = np.stack(df.apply(quantum_encode, axis=1))
y = df[[f"bola{i}" for i in range(1, 7)]].mean(axis=1) // 10

print(f"✅ Total registros después del filtrado: {len(X)}")

# 🔀 División de datos (solo 200 muestras)
print("🔀 Seleccionando 200 muestras para entrenamiento...")
X_train, _, y_train, _ = train_test_split(X, y, train_size=200, random_state=42)
print(f"📦 Tamaño del conjunto de entrenamiento: {len(X_train)} muestras")

# ⚛️ Inicializar el modelo cuántico
print("⚛️ Inicializando QSVC con QuantumKernel...")
feature_map = ZZFeatureMap(feature_dimension=50, reps=1)
kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('aer_simulator'))
model = QSVC(quantum_kernel=kernel)

# 🧠 Entrenamiento del modelo
print("🚀 Entrenando el modelo cuántico... (esto puede tardar unos minutos ⏳)")
start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f"✅ Entrenamiento completado en {round(end - start, 2)} segundos.")

# 💾 Guardar modelo entrenado
print("💾 Guardando el modelo en 'modelo_qsvc_tinka.joblib'...")
dump(model, "modelo_qsvc_tinka.joblib")
print("✅ Modelo cuántico entrenado y guardado con éxito.")
