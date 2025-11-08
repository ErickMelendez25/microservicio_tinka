# main.py
import os
import numpy as np
import joblib
import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.utils import algorithm_globals
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks

app = FastAPI()
load_dotenv()


# =============================
# 🔒 Configuración de CORS
# =============================
origins = [
    "http://localhost:3000",
    "https://tinka-production.up.railway.app",
    "https://tinka.grupo-digital-nextri.com"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],        # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],        # Permitir todas las cabeceras
)

# =============================
# 🔹 Configuración de la base
# =============================
DB_HOST = os.getenv("DB_HOST", "gondola.proxy.rlwy.net")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_PORT = int(os.getenv("DB_PORT", 34954))
DB_NAME = os.getenv("DB_NAME", "railway")

def conectar_db():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        database=DB_NAME
    )

# ==============================================================
# 🔹 Generación de combinaciones con verificación de probabilidades
# ==============================================================
def generar_combinaciones_probabilisticas(frequencies, n=1000):
    """
    Genera un conjunto (n) de combinaciones candidatas ponderadas por frecuencia.
    Devuelve lista de arreglos shape (6,).
    """
    # Asegurar que frequencies tenga mínimo tamaño 48
    if len(frequencies) < 48:
        frequencies = np.pad(frequencies, (0, 48 - len(frequencies)))

    bolas = np.arange(1, len(frequencies) + 1)
    probs = frequencies / np.sum(frequencies)

    # Verificación de tamaño correcto
    if len(bolas) != len(probs):
        print(f"⚠️ Tamaños desiguales: bolas={len(bolas)}, probs={len(probs)} — ajustando...")
        min_len = min(len(bolas), len(probs))
        bolas = bolas[:min_len]
        probs = probs[:min_len]
        probs /= np.sum(probs)

    candidatos = []
    for _ in range(n):
        try:
            comb = np.random.choice(bolas, size=6, replace=False, p=probs)
            comb.sort()
            candidatos.append(comb)
        except Exception as e:
            print(f"Error al generar combinación: {e}")
            continue

    candidatos_np = np.unique(np.array(candidatos), axis=0)
    return candidatos_np

# ==============================================================
# 🔹 Puntuar candidatos con kernel cuántico
# ==============================================================
def score_candidates_with_kernel(candidates: np.ndarray, X_train_scaled: np.ndarray, kernel_obj) -> np.ndarray:
    if candidates.size == 0:
        return np.array([])

    try:
        K_cx = kernel_obj.evaluate(candidates, X_train_scaled)
        scores = np.mean(K_cx, axis=1)
        return scores
    except Exception as e:
        print("Error al puntuar con kernel:", e)
        freqs = np.bincount(candidates.flatten().astype(int), minlength=48)
        scores = np.mean(freqs) * np.ones(candidates.shape[0])
        return scores

# ==============================================================
# 🔹 Endpoint principal
# ==============================================================
@app.post("/api/ejecutarmodelos")
async def ejecutar_modelos(background_tasks: BackgroundTasks):
    background_tasks.add_task(_proceso_modelo)
    return {"status": "🧠 Modelo ejecutándose en segundo plano..."}

def _proceso_modelo():
    try:
        print("🔄 Iniciando ejecución del modelo en background...")
        conn = conectar_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT bola1, bola2, bola3, bola4, bola5, bola6
            FROM sorteos ORDER BY fecha DESC LIMIT 200
        """)
        rows = cursor.fetchall()
        X = np.array(rows, dtype=float)

        frequencies = np.bincount(X.flatten().astype(int), minlength=48)
        np.save("frequencies.npy", frequencies)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        fm = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
        kernel = FidelityQuantumKernel(feature_map=fm)

        candidatos = generar_combinaciones_probabilisticas(frequencies, n=1000)
        candidatos_scaled = scaler.transform(candidatos.astype(float))
        scores = score_candidates_with_kernel(candidatos_scaled, X_scaled, kernel)

        idx_sorted = np.argsort(scores)[::-1][:10]
        top = candidatos[idx_sorted]
        top_scores = scores[idx_sorted]

        for comb, score in zip(top, top_scores):
            cursor.execute("""
                INSERT INTO predicciones 
                (bola1, bola2, bola3, bola4, bola5, bola6, boliyapa, probabilidad, modelo_version)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                int(comb[0]), int(comb[1]), int(comb[2]),
                int(comb[3]), int(comb[4]), int(comb[5]),
                np.random.randint(1, 49), float(score), "QKernel_v2.1"
            ))
        conn.commit()
        conn.close()
        print("✅ Modelo finalizado y guardado en DB.")
    except Exception as e:
        print("❌ Error en proceso modelo:", e)
        
@app.get("/api/frecuencias")
def obtener_frecuencias():
    """
    Devuelve la frecuencia de aparición de cada número (1 al 48)
    según la tabla 'sorteos'.
    """
    try:
        conn = conectar_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bola1, bola2, bola3, bola4, bola5, bola6
            FROM sorteos
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"error": "No hay datos en la tabla 'sorteos'."}

        # Contar ocurrencias de cada número
        X = np.array(rows, dtype=int)
        frecuencias = np.bincount(X.flatten(), minlength=49)[1:]  # índices 1-48

        # Crear lista de diccionarios
        resultado = [
            {"numero": i + 1, "veces_salida": int(frecuencias[i])}
            for i in range(48)
        ]

        return resultado

    except Exception as e:
        print("Error en /api/frecuencias:", e)
        return {"error": str(e)}
