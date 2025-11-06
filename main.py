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

app = FastAPI()
load_dotenv()

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
def ejecutar_modelos(top_n: int = 10):
    """
    Genera combinaciones, las puntúa con el kernel cuántico
    y guarda las top_n en la tabla 'predicciones'.
    """
    try:
        print("Iniciando ejecución del modelo...")

        conn = conectar_db()
        cursor = conn.cursor()

        # 1) Leer históricos
        cursor.execute("""
            SELECT bola1, bola2, bola3, bola4, bola5, bola6
            FROM sorteos
            ORDER BY fecha DESC
            LIMIT 200
        """)
        rows = cursor.fetchall()
        if not rows:
            return {"error": "No hay registros en 'sorteos'."}
        X = np.array(rows, dtype=float)

        # 2) Cargar o generar frecuencias
        if os.path.exists("frequencies.npy"):
            frequencies = np.load("frequencies.npy")
            print("Frecuencias cargadas.")
        else:
            frequencies = np.bincount(X.flatten().astype(int), minlength=48)
            np.save("frequencies.npy", frequencies)
            print("Frecuencias calculadas y guardadas.")

        # 3) Cargar kernel y scaler
        fidelity_kernel = None
        scaler = None

        if os.path.exists("quantum_kernel_tinka.joblib"):
            fidelity_kernel = joblib.load("quantum_kernel_tinka.joblib")
            print("Kernel cuántico cargado.")
        else:
            algorithm_globals.random_seed = 42
            fm = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
            fidelity_kernel = FidelityQuantumKernel(feature_map=fm)
            joblib.dump(fidelity_kernel, "quantum_kernel_tinka.joblib")
            print("Kernel creado y guardado por defecto.")

        if os.path.exists("scaler_qsvc_tinka.joblib"):
            from sklearn.preprocessing import StandardScaler
            scaler = joblib.load("scaler_qsvc_tinka.joblib")
            print("Scaler cargado.")
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, "scaler_qsvc_tinka.joblib")
            print("Scaler creado y guardado.")

        X_scaled = scaler.transform(X)

        # 4) Generar candidatos
        candidatos = generar_combinaciones_probabilisticas(frequencies, n=3000)
        print("Candidatos generados:", candidatos.shape[0])

        if candidatos.size == 0:
            return {"error": "No se generaron candidatos válidos."}

        candidatos_scaled = scaler.transform(candidatos.astype(float))

        # 5) Puntuar
        scores = score_candidates_with_kernel(candidatos_scaled, X_scaled, fidelity_kernel)

        if len(scores) != len(candidatos):
            print(f"⚠️ Len(scores)={len(scores)} distinto de Len(candidatos)={len(candidatos)} — corrigiendo tamaños.")
            min_len = min(len(scores), len(candidatos))
            scores = scores[:min_len]
            candidatos = candidatos[:min_len]

        idx_sorted = np.argsort(scores)[::-1]
        top_idx = idx_sorted[:top_n]
        top_candidates = candidatos[top_idx]
        top_scores = scores[top_idx]

        # 6) Guardar top N
        nuevas = []
        for comb, score in zip(top_candidates, top_scores):
            boliyapa = int(np.random.choice(np.arange(1, len(frequencies) + 1),
                                            p=frequencies / np.sum(frequencies)))
            prob = float(score)
            modelo_version = "QKernel_v2.1"

            pred = {
                "bola1": int(comb[0]), "bola2": int(comb[1]), "bola3": int(comb[2]),
                "bola4": int(comb[3]), "bola5": int(comb[4]), "bola6": int(comb[5]),
                "boliyapa": boliyapa, "probabilidad": prob, "modelo_version": modelo_version
            }
            nuevas.append(pred)

            cursor.execute("""
                INSERT INTO predicciones 
                (bola1, bola2, bola3, bola4, bola5, bola6, boliyapa, probabilidad, modelo_version)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (pred["bola1"], pred["bola2"], pred["bola3"], pred["bola4"], pred["bola5"], pred["bola6"],
                  pred["boliyapa"], pred["probabilidad"], pred["modelo_version"]))
        conn.commit()
        conn.close()

        print(f"Guardadas {len(nuevas)} predicciones en DB.")
        return {"detalle": "Modelo ejecutado", "predicciones": nuevas}

    except Exception as e:
        print("Error en /api/ejecutarmodelos:", e)
        return {"error": str(e)}
