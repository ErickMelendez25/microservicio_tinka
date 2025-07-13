from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel, QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import seaborn as sns
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

# Crear carpetas necesarias
os.makedirs("static", exist_ok=True)
os.makedirs("graficos", exist_ok=True)
os.makedirs("interpretaciones", exist_ok=True)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

class ZonaRequest(BaseModel):
    zona_id: int

class DummyRequest(BaseModel):
    ejecutar: bool = True

@app.get("/")
def root():
    return {"mensaje": "✅ API de Tinka está corriendo correctamente"}

@app.post("/ejecutarmodelo")
def ejecutar_modelo(data: ZonaRequest):
    zona_id = data.zona_id

    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT s.tipo AS tipo_sensor, l.valor
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.dispositivo_id = d.id
        JOIN sensores s ON d.sensor_id = s.id
        WHERE d.zona_id = %s
    """, (zona_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    datos = {}
    for row in rows:
        tipo = row['tipo_sensor'].lower()
        valor = float(row['valor'])
        datos.setdefault(tipo, []).append(valor)

    tipos = ['temperatura', 'humedad', 'ph', 'nitrógeno', 'fósforo', 'potasio']
    num = len(datos.get('temperatura', []))
    if num == 0:
        return {"error": "No hay datos suficientes para ejecutar el modelo."}

    X = [[datos.get(tipo, [0]*num)[i] for tipo in tipos] for i in range(num)]
    X = StandardScaler().fit_transform(np.array(X))

    feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    matrix = kernel.evaluate(x_vec=X)

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
    labels = clustering.fit_predict(matrix)

    graficar_clusters(X, labels, zona_id)
    graficar_matriz_kernel(matrix, zona_id)
    generar_interpretacion(zona_id, labels, matrix, tipos)

    return {
        "zona_id": zona_id,
        "clusters": labels.tolist(),
        "tipos": tipos,
        "valores": X.tolist(),
        "imagenes": {
            "clusters": f"/grafico/cluster/{zona_id}",
            "kernel": f"/grafico/kernel/{zona_id}"
        },
        "interpretacion": f"/interpretacion/{zona_id}"
    }

@app.post("/api/ejecutarmodelos")
def ejecutar_modelo_loteria(request: DummyRequest):
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
        df = pd.DataFrame(cursor.fetchall())

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
        y = np.array([int(sum(row[[f'bola{i}' for i in range(1, 7)]]) / 6) // 10 for _, row in df.iterrows()])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        feature_map = ZZFeatureMap(feature_dimension=50, reps=2, entanglement='linear')
        quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('aer_simulator'))
        model = QSVC(quantum_kernel=quantum_kernel)
        model.fit(X_train, y_train)

        top_numeros = [n for n, _ in frecuencia_individual.most_common(30)]
        seleccionadas = random.sample(list(combinations(top_numeros, 6)), 10)

        cursor.execute("DELETE FROM predicciones")
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

            cursor.execute("""
                INSERT INTO predicciones (
                    bola1, bola2, bola3, bola4, bola5, bola6,
                    boliyapa, probabilidad, modelo_version, pares, trios
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(bolas) + (boliyapa, probabilidad, "Qiskit-v1", top_pares, top_trios))

            predicciones.append({**row, "probabilidad": probabilidad, "modelo_version": "Qiskit-v1", "pares": top_pares, "trios": top_trios})

        # Visualización cuántica
        qc_viz = QuantumCircuit(3, 3)
        qc_viz.h([0, 1, 2])
        qc_viz.measure([0, 1, 2], [0, 1, 2])
        sim = Aer.get_backend('aer_simulator')
        conteos = sim.run(qc_viz, shots=5000).result().get_counts()

        conteos_dado = {int(k, 2): v for k, v in conteos.items() if 1 <= int(k, 2) <= 6}
        plot_histogram(conteos_dado, title="Visualización Cuántica: Superposición y Colapso")
        plt.savefig("static/superposicion_colapso.png")
        plt.close()

        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "✅ Modelo ejecutado correctamente", "predicciones_generadas": predicciones}

    except Exception as e:
        return {"error": str(e)}

@app.get("/grafico/cluster/{zona_id}")
def ver_cluster(zona_id: int):
    return FileResponse(os.path.join("graficos", f"clusters_zona_{zona_id}.png"), media_type="image/png")

@app.get("/grafico/kernel/{zona_id}")
def ver_kernel(zona_id: int):
    return FileResponse(os.path.join("graficos", f"kernel_zona_{zona_id}.png"), media_type="image/png")

@app.get("/interpretacion/{zona_id}")
def ver_interpretacion(zona_id: int):
    return FileResponse(os.path.join("interpretaciones", f"interpretacion_zona_{zona_id}.txt"), media_type="text/plain")

def graficar_clusters(X, labels, zona_id):
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"Clústeres - Zona {zona_id}")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig(f"graficos/clusters_zona_{zona_id}.png")
    plt.close()

def graficar_matriz_kernel(matrix, zona_id):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, cmap='coolwarm', square=True, cbar=True)
    plt.title(f"Matriz Kernel - Zona {zona_id}")
    plt.savefig(f"graficos/kernel_zona_{zona_id}.png")
    plt.close()

def generar_interpretacion(zona_id, labels, matrix, tipos):
    total = len(labels)
    grupo_0 = sum(1 for l in labels if l == 0)
    grupo_1 = total - grupo_0
    texto = f"""
INTERPRETACIÓN DE RESULTADOS - ZONA {zona_id}

Se identificaron 2 grupos:
- Grupo 0: {grupo_0} muestras ({(grupo_0 / total) * 100:.1f}%)
- Grupo 1: {grupo_1} muestras ({(grupo_1 / total) * 100:.1f}%)

Variables analizadas: {', '.join(tipos)}.
"""
    with open(f"interpretaciones/interpretacion_zona_{zona_id}.txt", "w", encoding="utf-8") as f:
        f.write(texto.strip())
