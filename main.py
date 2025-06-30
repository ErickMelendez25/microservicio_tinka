# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
import mysql.connector
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o especifica tu dominio de frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ZonaRequest(BaseModel):
    zona_id: int

@app.post("/ejecutarmodelo")
def ejecutar_modelo(data: ZonaRequest):
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
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

    return {
        "zona_id": zona_id,
        "clusters": labels.tolist(),
        "tipos": tipos,
        "valores": X.tolist()
    }
