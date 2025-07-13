import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# --- Cargar variables de entorno ---
load_dotenv()

# --- Conectar a la base de datos ---
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT", 3306))
)
cursor = conn.cursor(dictionary=True)

# --- 1. Obtener datos de sorteos históricos ---
cursor.execute("SELECT * FROM sorteos ORDER BY fecha ASC")
sorteos = cursor.fetchall()
df = pd.DataFrame(sorteos)

# --- 2. Obtener tabla de frecuencias ---
cursor.execute("SELECT * FROM frecuencias")
frecuencias = pd.DataFrame(cursor.fetchall())
frecuencia_dict = frecuencias.set_index('numero')['veces_salida'].to_dict()

# --- 3. Codificación cuántica: combinación a vector de 51 dimensiones ---
def encode_combination(row):
    vec = np.zeros(51)  # 50 bolas + 1 para boliyapa
    for b in ['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 'bola6']:
        vec[int(row[b]) - 1] = 1
    if row['boliyapa']:
        vec[int(row['boliyapa']) - 1] += 0.5  # peso menor a boliyapa
    return vec

X = np.stack(df.apply(encode_combination, axis=1))

# --- 4. Superposición cuántica simulada: PCA después de escalar ---
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=3).fit_transform(X_scaled)

# --- 5. Colapso cuántico simulado: Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca)
centroids = kmeans.cluster_centers_
distances = euclidean_distances(X_pca, centroids)

# --- 6. Seleccionar combinaciones más cercanas a los centroides ---
closest_points = np.argmin(distances, axis=0)

print("🔮 Combinaciones predichas más probables:")

for i, idx in enumerate(closest_points):
    bolas = df.loc[idx, ['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 'bola6']].values.tolist()
    boliyapa = df.loc[idx, 'boliyapa']
    
    # Aseguramos tipos int para insertar en MySQL
    bolas = [int(b) for b in bolas]
    boliyapa = int(boliyapa) if boliyapa else None

    # Ordenar por frecuencia: más comunes primero
    bolas.sort(key=lambda x: -frecuencia_dict.get(x, 0))

    # Calcular probabilidad simulada (inversamente proporcional a la distancia)
    dist = distances[idx, i]
    prob = max(0.01, min(1.0, 1 / (1 + dist)))

    print(f"👉 Combinación sugerida: {bolas} + BoliYapa: {boliyapa}")

    # --- 7. Insertar predicción en la base de datos ---
    cursor.execute("""
        INSERT INTO predicciones (
            probabilidad, bola1, bola2, bola3, bola4, bola5, bola6, boliyapa, modelo_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        float(prob),
        bolas[0], bolas[1], bolas[2],
        bolas[3], bolas[4], bolas[5],
        boliyapa,
        "Qiskit-v1"
    ))

# Finalizar transacción y cerrar conexiones
conn.commit()
cursor.close()
conn.close()
print("✅ Predicciones guardadas correctamente.")
