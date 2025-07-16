from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from itertools import combinations
import random
from collections import Counter

# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
                             #allow_origins=["https://tinka.academionlinegpt.com","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- solo durante desarrollo
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Crear carpeta static si no existe
if not os.path.exists("static"):
    os.makedirs("static")

# Esquema de request
class DummyRequest(BaseModel):
    ejecutar: bool = True

@app.get("/")
def root():
    return {"mensaje": "✅ API de Tinka está corriendo correctamente"}

@app.post("/api/ejecutarmodelo")
def ejecutar_modelo(request: DummyRequest):
    try:
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
        df = pd.DataFrame(sorteos)

        if len(df) < 8:
            return {"error": "❌ No hay suficientes datos para entrenar el modelo."}

        todas_bolas = df[[f'bola{i}' for i in range(1, 7)]].values.flatten()
        frecuencia_individual = Counter(todas_bolas)

        pares = Counter()
        trios = Counter()
        for _, row in df.iterrows():
            bolas = [int(row[f'bola{i}']) for i in range(1, 7)]
            pares.update(combinations(sorted(bolas), 2))
            trios.update(combinations(sorted(bolas), 3))

        def quantum_encode(row):
            vec = np.zeros(50)
            for b in ['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 'bola6']:
                vec[int(row[b]) - 1] = 1
            try:
                boliyapa = int(row['boliyapa'])
                if boliyapa not in [int(row[f'bola{i}']) for i in range(1, 7)]:
                    vec[boliyapa - 1] += 0.5
            except:
                pass
            return vec

        X = np.stack(df.apply(quantum_encode, axis=1))
        y = np.array([
            int(sum(row[[f'bola{i}' for i in range(1, 7)]]) / 6) // 10
            for _, row in df.iterrows()
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        feature_map = ZZFeatureMap(feature_dimension=50, reps=2, entanglement='linear')
        quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('aer_simulator'))
        model = QSVC(quantum_kernel=quantum_kernel)
        model.fit(X_train, y_train)

        top_numeros = [n for n, _ in frecuencia_individual.most_common(30)]
        mejores_combos = list(combinations(top_numeros, 6))
        random.shuffle(mejores_combos)
        seleccionadas = mejores_combos[:10]

        print(f"🔄 Se generarán {len(seleccionadas)} predicciones...")

        cursor.execute("DELETE FROM predicciones")
        print("🗑️ Tabla 'predicciones' limpiada.")

        predicciones = []

        for bolas in seleccionadas:
            bolas = sorted(bolas)
            boliyapa = random.choice([n for n in range(1, 51) if n not in bolas])
            row = {f'bola{i+1}': bolas[i] for i in range(6)}
            row['boliyapa'] = boliyapa
            vec = quantum_encode(row)
            pred = model.predict([vec])[0]
            media = sum(bolas) // 6
            clase_real = media // 10
            probabilidad = 1.0 if pred == clase_real else 0.5
            top_pares = sum([pares[p] for p in combinations(bolas, 2)])
            top_trios = sum([trios[t] for t in combinations(bolas, 3)])

            print(f"🔢 Insertando: {bolas} + BY:{boliyapa} Prob:{probabilidad} Pares:{top_pares} Trios:{top_trios}")
            
            cursor.execute("""
                INSERT INTO predicciones (
                    bola1, bola2, bola3, bola4, bola5, bola6,
                    boliyapa, probabilidad, modelo_version, pares, trios
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, 
            tuple(map(int, bolas)) + (
                int(boliyapa),
                float(probabilidad),
                "Qiskit-v1",
                int(top_pares),
                int(top_trios)
            ))

            predicciones.append({
                "bola1": int(bolas[0]),
                "bola2": int(bolas[1]),
                "bola3": int(bolas[2]),
                "bola4": int(bolas[3]),
                "bola5": int(bolas[4]),
                "bola6": int(bolas[5]),
                "boliyapa": int(boliyapa),
                "probabilidad": float(probabilidad),
                "modelo_version": "Qiskit-v1",
                "pares": int(top_pares),
                "trios": int(top_trios)
            })

        # -------- VISUALIZACIÓN CUÁNTICA -----------
        qc_viz = QuantumCircuit(3, 3)
        qc_viz.h([0, 1, 2])
        qc_viz.measure([0, 1, 2], [0, 1, 2])
        sim = Aer.get_backend('aer_simulator')
        resultado_viz = sim.run(qc_viz, shots=5000).result()
        conteos_viz = resultado_viz.get_counts()

        conteos_dado = {}
        for estado, frecuencia in conteos_viz.items():
            valor = int(estado, 2)
            if 1 <= valor <= 6:
                conteos_dado[valor] = conteos_dado.get(valor, 0) + frecuencia

        plt.figure(figsize=(6, 4))
        plot_histogram(conteos_dado, title="Visualización Cuántica: Superposición y Colapso")
        plt.savefig("static/superposicion_colapso.png")
        plt.close()
        # ----------------------------------------------

        conn.commit()
        print(f"✅ {len(predicciones)} predicciones insertadas correctamente en la base de datos.")

        cursor.close()
        conn.close()

        return {
            "message": "✅ Modelo ejecutado correctamente",
            "predicciones_generadas": predicciones
        }

    except Exception as e:
        print(f"❌ Error durante ejecución del modelo: {str(e)}")
        return {"error": str(e)}
