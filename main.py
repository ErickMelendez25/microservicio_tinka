# main.py
import os
import numpy as np
import joblib
import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List, Dict
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.utils import algorithm_globals

app = FastAPI()
load_dotenv()

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

def generar_combinaciones_probabilisticas(frequencies, n=1000):
    """
    Genera un conjunto (n) de combinaciones candidatas ponderadas por frecuencia.
    Devuelve lista de arreglos shape (6,).
    """
    bolas = np.arange(1, len(frequencies))
    probs = frequencies / np.sum(frequencies)
    candidatos = []
    # generamos n candidatos, cada uno sampleando 6 bolas sin reemplazo con probs
    for _ in range(n):
        comb = np.random.choice(bolas, size=6, replace=False, p=probs)
        comb.sort()
        candidatos.append(comb)
    # devolver array unico (eliminar duplicados)
    candidatos_np = np.unique(np.array(candidatos), axis=0)
    return candidatos_np

def score_candidates_with_kernel(candidates: np.ndarray, X_train_scaled: np.ndarray, kernel_obj) -> np.ndarray:
    """
    Puntúa cada candidato por su similitud promedio con el set de entrenamiento
    usando el objeto fidelity kernel (su método evaluate).
    """
    if candidates.size == 0:
        return np.array([])
    # candidates shape (M,6). Escalar con scaler si fue usado (aquí asumimos scaler ya aplicado por quien genera candidatos)
    # Kernel expects input feature vectors in same shape as training.
    try:
        K_cx = kernel_obj.evaluate(candidates, X_train_scaled)  # shape (M, Ntrain)
        scores = np.mean(K_cx, axis=1)  # similitud promedio con datos reales
        return scores
    except Exception as e:
        print("Error al puntuar con kernel:", e)
        # fallback: usar frecuencia media de bolas como proxy
        freqs = np.bincount(candidates.flatten().astype(int), minlength=48)
        scores = np.mean(freqs) * np.ones(candidates.shape[0])
        return scores

@app.post("/api/ejecutarmodelos")
def ejecutar_modelos(top_n: int = 10):
    """
    Endpoint que genera combinaciones, las puntúa con el kernel cuántico
    y guarda las top_n en la tabla 'predicciones'.
    """
    try:
        print("Iniciando ejecución del modelo...")

        conn = conectar_db()
        cursor = conn.cursor()

        # 1) Leer historiales (si necesitas X_train_scaled para kernel)
        cursor.execute("""
            SELECT bola1, bola2, bola3, bola4, bola5, bola6
            FROM sorteos
            ORDER BY fecha DESC
            LIMIT 200
        """)
        rows = cursor.fetchall()
        if not rows:
            return {"error": "No hay registros en 'sorteos'."}
        X = np.array(rows, dtype=float)  # shape (N,6)

        # 2) Cargar frecuencias
        if os.path.exists("frequencies.npy"):
            frequencies = np.load("frequencies.npy")
            print("Frecuencias cargadas.")
        else:
            frequencies = np.bincount(X.flatten().astype(int), minlength=48)
            np.save("frequencies.npy", frequencies)
            print("Frecuencias calculadas y guardadas.")

        # 3) Cargar kernel y scaler si existen
        fidelity_kernel = None
        scaler = None
        if os.path.exists("quantum_kernel_tinka.joblib"):
            fidelity_kernel = joblib.load("quantum_kernel_tinka.joblib")
            print("Kernel cuántico cargado.")
        else:
            # crear un kernel sencillo si no existe
            algorithm_globals.random_seed = 42
            fm = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
            fidelity_kernel = FidelityQuantumKernel(feature_map=fm)
            joblib.dump(fidelity_kernel, "quantum_kernel_tinka.joblib")
            print("Kernel no existente — creado y guardado por defecto.")

        if os.path.exists("scaler_qsvc_tinka.joblib"):
            scaler = joblib.load("scaler_qsvc_tinka.joblib")
            print("Scaler cargado.")
        else:
            # crear scaler básico y guardarlo
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, "scaler_qsvc_tinka.joblib")
            print("Scaler no existente — creado y guardado.")

        # aplicar escala a X para que kernel y scoring sean coherentes
        X_scaled = scaler.transform(X)

        # 4) Generar muchos candidatos (aleatorios ponderados por frecuencia)
        candidatos = generar_combinaciones_probabilisticas(frequencies, n=3000)  # genera ~3000 candidatos
        print("Candidatos generados:", candidatos.shape[0])

        # Escalar candidatos con el mismo scaler
        candidatos_scaled = scaler.transform(candidatos.astype(float))

        # 5) Puntuar candidatos con kernel
        scores = score_candidates_with_kernel(candidatos_scaled, X_scaled, fidelity_kernel)
        # combinamos candidatos y scores
        idx_sorted = np.argsort(scores)[::-1]  # descendente
        top_idx = idx_sorted[:top_n]
        top_candidates = candidatos[top_idx]
        top_scores = scores[top_idx]

        # 6) Preparar y guardar top N en la tabla predicciones
        nuevas = []
        for comb, score in zip(top_candidates, top_scores):
            # generar boliyapa ponderado por frecuencia
            boliyapa = int(np.random.choice(np.arange(1, len(frequencies)), p=frequencies/np.sum(frequencies)))
            prob = float(score)  # la "probabilidad" la representamos por el score (normalizar si quieres)
            modelo_version = "QKernel_v2.1"

            pred = {
                "bola1": int(comb[0]), "bola2": int(comb[1]), "bola3": int(comb[2]),
                "bola4": int(comb[3]), "bola5": int(comb[4]), "bola6": int(comb[5]),
                "boliyapa": boliyapa, "probabilidad": prob, "modelo_version": modelo_version
            }
            nuevas.append(pred)

            # Insertar en la tabla predicciones
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
