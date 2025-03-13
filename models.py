import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = np.array([[d["habilidade"], d["tempo_jogo"], d["familiaridade"], d["competitividade"], d["imersao"]] for d in data])
y = np.array([d["categoria"] for d in data])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  

model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

print("📌 Modelo supervisionado treinado com sucesso!\n")
print("Coeficientes (pesos de cada atributo):", model.coef_)
print("Intercepto (ajuste inicial da equação):", model.intercept_)

kmeans = KMeans(n_clusters=5, random_state=42) 
kmeans.fit(X)

print("\n📊 Centros dos clusters (médias dos atributos em cada grupo):")
print(kmeans.cluster_centers_)

labels_kmeans = kmeans.predict(X)

print("\n📊 Atribuição de jogadores aos grupos (clusters):")
for i, d in enumerate(data):
    print(f"Jogador {i+1} pertence ao grupo {labels_kmeans[i]}: {d}")

def prever_categoria(habilidade, tempo_jogo, familiaridade, competitividade, imersao):
    try:
        entrada = np.array([[habilidade, tempo_jogo, familiaridade, competitividade, imersao]])

        print("\n🔍 Passo a passo da previsão:")
        print(f"🎮 Entrada do jogador: Habilidade={habilidade}, Tempo de Jogo={tempo_jogo}, Familiaridade={familiaridade}, Competitividade={competitividade}, Imersão={imersao}")

        predicao_numerica = model.predict(entrada)[0]
        print(f"📊 Resultado da classificação (Regressão Logística): {predicao_numerica}")

        categoria_prevista = label_encoder.inverse_transform([predicao_numerica])[0]
        print(f"🎯 Categoria prevista (Regressão Logística): {categoria_prevista}")

        return categoria_prevista

    except Exception as e:
        print(f"❌ Erro ao realizar a previsão: {e}")
        return None

novo_jogador = {"habilidade": 75, "tempo_jogo": 1200, "familiaridade": 80, "competitividade": 90, "imersao": 60}
categoria_supervisionada = prever_categoria(**novo_jogador)

novo_jogador_kmeans = np.array([[75, 1200, 80, 90, 60]]) 
cluster_novo_jogador = kmeans.predict(novo_jogador_kmeans)
print(f"O novo jogador pertence ao grupo {cluster_novo_jogador[0]} (K-Means)")

print("\n✅ Previsão Final (Regressão Logística):", categoria_supervisionada)
