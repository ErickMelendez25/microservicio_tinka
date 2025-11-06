import os
import mysql.connector
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler

# ======================
# Cargar .env
# ======================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "gondola.proxy.rlwy.net")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "UdeLsMmVgolwytbCQvfTEJbhoMHpLdOz")
DB_PORT = int(os.getenv("DB_PORT", 34954))
DB_NAME = os.getenv("DB_NAME", "railway")

# Debug para confirmar que las variables se leen
print("DEBUG DB_HOST:", DB_HOST)
print("DEBUG DB_USER:", DB_USER)
print("DEBUG DB_PASSWORD:", "(oculto)" if DB_PASSWORD else None)
print("DEBUG DB_PORT:", DB_PORT)
print("DEBUG DB_NAME:", DB_NAME)

# ======================
# Conectar a DB y traer los últimos 200 registros
# ======================
print("🔎 Leyendo historiales desde DB (últimos 200 registros)...")

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
    raise ValueError("⚠️ No hay registros en la tabla 'sorteos'.")

# ======================
# Preparar datos
# ======================
print("🔢 Preparando X, y...")

X = np.array(rows)
y = np.arange(len(rows)) % 2  # etiquetas dummy (0 y 1) solo para que entrene

# ======================
# Estadísticas simples (frecuencia de bolas)
# ======================
print("📊 Calculando estadísticas de frecuencia...")

frequencies = np.bincount(X.flatten(), minlength=48)  # soporta hasta bola 47
top_balls = np.argsort(frequencies)[::-1][:10]

print(f"🎯 Bolas más frecuentes: {top_balls}")

# ======================
# Escalar los datos
# ======================
print("⚖️ Escalando X con StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_qsvc_tinka.joblib")
print("✅ Scaler guardado: scaler_qsvc_tinka.joblib")

# ======================
# Crear Quantum Kernel
# ======================
print("🎛️ Entrenando modelo cuántico (kernel) con 200 registros...")

algorithm_globals.random_seed = 12345
feature_map = ZZFeatureMap(feature_dimension=X_scaled.shape[1], reps=2, entanglement="linear")

sampler = Sampler()
fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)

print("✅ Creando FidelityQuantumKernel (esto puede tardar un poco)...")

kernel_matrix = fidelity_kernel.evaluate(X_scaled)
print("✅ Matriz kernel calculada con shape:", kernel_matrix.shape)

# ======================
# Guardar kernel y estadísticas
# ======================
joblib.dump(fidelity_kernel, "quantum_kernel_tinka.joblib")
print("✅ Kernel cuántico guardado: quantum_kernel_tinka.joblib")

np.save("frequencies.npy", frequencies)
print("✅ Frecuencias guardadas en frequencies.npy")

print("🏁 Entrenamiento terminado con éxito usando 200 registros")
