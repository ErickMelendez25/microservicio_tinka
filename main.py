# main.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import mysql.connector
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qiskit.academionlinegpt.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios de salida
GRAFICOS_DIR = "graficos"
INTERPRETACIONES_DIR = "interpretaciones"
os.makedirs(GRAFICOS_DIR, exist_ok=True)
os.makedirs(INTERPRETACIONES_DIR, exist_ok=True)

class ZonaRequest(BaseModel):
    zona_id: int

# Gráfico de clusters
def graficar_clusters(X, labels, zona_id):
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"Clústeres - Zona {zona_id}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    path = os.path.join(GRAFICOS_DIR, f"clusters_zona_{zona_id}.png")
    plt.savefig(path)
    plt.close()

# Gráfico matriz kernel
def graficar_matriz_kernel(matrix, zona_id):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, cmap='coolwarm', square=True, cbar=True)
    plt.title(f"Matriz Kernel - Zona {zona_id}")
    plt.xlabel("Índice")
    plt.ylabel("Índice")
    path = os.path.join(GRAFICOS_DIR, f"kernel_zona_{zona_id}.png")
    plt.savefig(path)
    plt.close()

# Interpretación automática
def generar_interpretacion(zona_id, labels, matrix, tipos):
    total = len(labels)
    grupo_0 = sum(1 for l in labels if l == 0)
    grupo_1 = total - grupo_0

    explicacion = f"""
INTERPRETACIÓN DE RESULTADOS - ZONA {zona_id}

🔹 CLÚSTERES:
Se identificaron 2 grupos principales utilizando clustering cuántico:
- Grupo 0: {grupo_0} muestras ({(grupo_0 / total) * 100:.1f}%)
- Grupo 1: {grupo_1} muestras ({(grupo_1 / total) * 100:.1f}%)

Estos grupos se formaron según variables como: {", ".join(tipos)}.

➤ Un gráfico generado (clusters_zona_{zona_id}.png) muestra cómo estas lecturas se agrupan según sus características usando un análisis cuántico de similitud.

🔹 MATRIZ KERNEL CUÁNTICA:
La matriz kernel (kernel_zona_{zona_id}.png) representa la similitud cuántica entre lecturas:
- Colores más claros: alta similitud
- Colores oscuros: baja similitud

Si ves bloques bien definidos en la matriz, eso indica que las lecturas tienen comportamientos distintos y el modelo cuántico detectó patrones claros.

✅ Este modelo se basa en estados cuánticos generados por cada muestra mediante un ZZFeatureMap, y mide la similitud con Fidelity Quantum Kernel de Qiskit.

💡 Con esto, puedes tomar decisiones informadas sobre manejo diferenciado del suelo en la zona.
    """

    path = os.path.join(INTERPRETACIONES_DIR, f"interpretacion_zona_{zona_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(explicacion.strip())

# Endpoint principal
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

    X = []
    for i in range(num):
        fila = [datos.get(tipo, [0]*num)[i] for tipo in tipos]
        X.append(fila)

    X = np.array(X)
    X = StandardScaler().fit_transform(X)

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

# Imagen de clúster
@app.get("/grafico/cluster/{zona_id}")
def ver_cluster(zona_id: int):
    path = os.path.join(GRAFICOS_DIR, f"clusters_zona_{zona_id}.png")
    return FileResponse(path, media_type="image/png")

# Imagen kernel
@app.get("/grafico/kernel/{zona_id}")
def ver_kernel(zona_id: int):
    path = os.path.join(GRAFICOS_DIR, f"kernel_zona_{zona_id}.png")
    return FileResponse(path, media_type="image/png")

# Endpoint de interpretación
@app.get("/interpretacion/{zona_id}")
def ver_interpretacion(zona_id: int):
    path = os.path.join(INTERPRETACIONES_DIR, f"interpretacion_zona_{zona_id}.txt")
    return FileResponse(path, media_type="text/plain", filename=f"interpretacion_zona_{zona_id}.txt")
