# train_model.py
import os
import mysql.connector
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals

# ======================
# Cargar .env
# ======================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "gondola.proxy.rlwy.net")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_PORT = int(os.getenv("DB_PORT", 34954))
DB_NAME = os.getenv("DB_NAME", "railway")

print("DEBUG: Conectando a DB:", DB_HOST, DB_NAME)

# ======================
# Conectar y leer DB
# ======================
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT,
    database=DB_NAME
)
cursor = conn.cursor()
cursor.execute("""
    SELECT bola1, bola2, bola3, bola4, bola5, bola6
    FROM sorteos
    ORDER BY fecha DESC
    LIMIT 200
""")
rows = cursor.fetchall()
conn.close()

if not rows:
    raise ValueError("No hay registros en la tabla 'sorteos'.")

# ======================
# Preparar datos
# ======================
X = np.array(rows, dtype=float)  # shape (N,6)
print("Registros cargados:", X.shape)

# ======================
# Estadísticas (frecuencia)
# ======================
frequencies = np.bincount(X.flatten().astype(int), minlength=48)  # soporte hasta 47
top_balls = np.argsort(frequencies)[::-1][:10]
print("Top bolas (freq):", top_balls)

# Guardar frecuencias
np.save("frequencies.npy", frequencies)
print("Frecuencias guardadas en frequencies.npy")

# ======================
# Escalado
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_qsvc_tinka.joblib")
print("Scaler guardado: scaler_qsvc_tinka.joblib")

# ======================
# Crear y evaluar Fidelity Quantum Kernel
# ======================
algorithm_globals.random_seed = 42
feature_map = ZZFeatureMap(feature_dimension=X_scaled.shape[1], reps=2, entanglement="linear")

print("Creando FidelityQuantumKernel (esto puede tardar)...")
fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map)  # sin sampler explícito

# calcular la matriz kernel (esto es costoso pero se guarda)
kernel_matrix = fidelity_kernel.evaluate(X_scaled)
print("Matriz kernel calculada shape:", kernel_matrix.shape)

# Guardar el kernel (objeto) para que main.py lo pueda cargar
joblib.dump(fidelity_kernel, "quantum_kernel_tinka.joblib")
print("Kernel guardado: quantum_kernel_tinka.joblib")

print("Entrenamiento completado.")
