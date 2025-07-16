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
from joblib import load

# Cargar variables de entorno
load_dotenv()

# 🔧 Inicializar FastAPI
app = FastAPI()

# ✅ CORS middleware (🔒 muy importante que esté justo después de app = FastAPI())


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origins=["https://tinka.academionlinegpt.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# 📁 Crear carpetas necesarias
os.makedirs("static", exist_ok=True)
os.makedirs("graficos", exist_ok=True)
os.makedirs("interpretaciones", exist_ok=True)

# 🖼️ Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


# 🧾 Modelos
class ZonaRequest(BaseModel):
    zona_id: int

class DummyRequest(BaseModel):
    ejecutar: bool = True


# ✅ Endpoint de prueba
@app.get("/")
def root():
    return {"mensaje": "✅ API de Tinka está corriendo correctamente"}

@app.options("/api/ejecutarmodelos")
def preflight_handler():
    return {"message": "Preflight received"}



# 🔍 MODELO 1: Análisis por zona
@app.post("/ejecut")
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


# 🔮 MODELO 2: Predicción cuántica de La Tinka
@app.post("/api/ejecutarmodelos")
def ejecutar_modelo_loteria(_: DummyRequest):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        cursor = conn.cursor(dictionary=True)

        # Traer sorteos y procesar datos
        cursor.execute("SELECT * FROM sorteos ORDER BY fecha ASC")
        sorteos = cursor.fetchall()
        df = pd.DataFrame(sorteos).dropna(subset=[f"bola{i}" for i in range(1, 7)])
        
        # Análisis estadístico
        todas_bolas = df[[f"bola{i}" for i in range(1, 7)]].values.flatten()
        frecuencia = Counter(todas_bolas)
        pares, trios = Counter(), Counter()
        for _, row in df.iterrows():
            bolas = [row[f"bola{i}"] for i in range(1, 7)]
            pares.update(combinations(sorted(bolas), 2))
            trios.update(combinations(sorted(bolas), 3))

        # Codificador directo (6 bolas → vector de 6 números)
        def quantum_encode(bolas):
            return [float(b) for b in bolas]  # <-- Compatible con el modelo entrenado y serialización

        # Cargar modelo
        model = load("modelo_qsvc_tinka.joblib")
        scaler = load("scaler_qsvc_tinka.joblib")

        # Generar combinaciones
        top_bolas = [n for n, _ in frecuencia.most_common(25)]
        combinaciones = random.sample(list(combinations(top_bolas, 6)), 100)

        # Eliminar predicciones anteriores
        cursor.execute("DELETE FROM predicciones")

        predicciones = []

        for bolas in combinaciones:
            bolas = sorted(bolas)
            vec = quantum_encode(bolas)
            vec_scaled = scaler.transform([vec])
            pred = model.predict(vec_scaled)[0]
            media = sum(bolas) / 6
            clase_real = media // 10
            probabilidad = 1.0 if pred == clase_real else (0.9 if abs(pred - clase_real) <= 1 else 0.5)

            top_pares = sum([pares[p] for p in combinations(bolas, 2)])
            top_trios = sum([trios[t] for t in combinations(bolas, 3)])
            boliyapa = random.choice([n for n in range(1, 51) if n not in bolas])

            # Conversión explícita a tipos nativos
            registro = {
                **{f"bola{i+1}": int(bolas[i]) for i in range(6)},
                "boliyapa": int(boliyapa),
                "probabilidad": float(round(probabilidad, 2)),
                "modelo_version": "Qiskit-v3",
                "pares": int(top_pares),
                "trios": int(top_trios)
            }

            cursor.execute("""
                INSERT INTO predicciones (
                    bola1, bola2, bola3, bola4, bola5, bola6,
                    boliyapa, probabilidad, modelo_version, pares, trios
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(registro.values()))
            
            predicciones.append(registro)

        conn.commit()

        # Retornar las 15 mejores combinaciones al frontend
        predicciones_ordenadas = sorted(
            predicciones,
            key=lambda x: (-x["probabilidad"], -x["pares"], -x["trios"])
        )[:15]

        return {
            "status": "✅ Predicciones generadas con modelo cuántico",
            "predicciones": predicciones_ordenadas
        }

    except Exception as e:
        if conn:
            conn.rollback()
        return {"error": str(e)}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



# 🖼️ Rutas para ver imágenes y textos
@app.get("/grafico/cluster/{zona_id}")
def ver_cluster(zona_id: int):
    return FileResponse(os.path.join("graficos", f"clusters_zona_{zona_id}.png"), media_type="image/png")

@app.get("/grafico/kernel/{zona_id}")
def ver_kernel(zona_id: int):
    return FileResponse(os.path.join("graficos", f"kernel_zona_{zona_id}.png"), media_type="image/png")

@app.get("/interpretacion/{zona_id}")
def ver_interpretacion(zona_id: int):
    return FileResponse(os.path.join("interpretaciones", f"interpretacion_zona_{zona_id}.txt"), media_type="text/plain")


# 📊 Funciones auxiliares
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
