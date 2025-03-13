from flask import Flask, request, jsonify
from flask_cors import CORS
from models import prever_categoria
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

CORS(app)

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = np.array([[d["habilidade"], d["tempo_jogo"], d["familiaridade"], d["competitividade"], d["imersao"]] for d in data])
y = np.array([d["categoria"] for d in data])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

def obter_clusters_data():
    clusters = {}
    for i, d in enumerate(data):
        cluster_id = int(kmeans.predict([X[i]])[0])
        categoria = label_encoder.inverse_transform([y_encoded[i]])[0] 
        if cluster_id not in clusters:
            clusters[cluster_id] = {"jogadores": [], "centro": kmeans.cluster_centers_[cluster_id].tolist()}
        clusters[cluster_id]["jogadores"].append({
            "id": i + 1,
            "habilidade": int(d["habilidade"]), 
            "tempo_jogo": int(d["tempo_jogo"]), 
            "familiaridade": int(d["familiaridade"]),  
            "competitividade": int(d["competitividade"]),
            "imersao": int(d["imersao"]), 
            "categoria": categoria
        })
    return clusters

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Classificação de Gamers está online!"})

@app.route("/prever", methods=["POST"])
def prever():
    try:
        dados = request.get_json()

        atributos_necessarios = ["habilidade", "tempo_jogo", "familiaridade", "competitividade", "imersao"]
        if not all(attr in dados for attr in atributos_necessarios):
            return jsonify({"erro": "Faltam atributos na requisição!"}), 400

        categoria = prever_categoria(
            habilidade=dados["habilidade"],
            tempo_jogo=dados["tempo_jogo"],
            familiaridade=dados["familiaridade"],
            competitividade=dados["competitividade"],
            imersao=dados["imersao"]
        )

        return jsonify({"categoria_prevista": categoria})

    except Exception as e:
        print(f"Erro ao processar a requisição: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.route("/clusters", methods=["GET"])
def obter_clusters():
    clusters = obter_clusters_data()

    for cluster_id, cluster_data in clusters.items():

        cluster_data["centro"] = [float(x) for x in cluster_data["centro"]] if isinstance(cluster_data["centro"], tuple) else cluster_data["centro"]
        
        for jogador in cluster_data["jogadores"]:
            jogador["habilidade"] = int(jogador["habilidade"])
            jogador["tempo_jogo"] = int(jogador["tempo_jogo"])
            jogador["familiaridade"] = int(jogador["familiaridade"])
            jogador["competitividade"] = int(jogador["competitividade"])
            jogador["imersao"] = int(jogador["imersao"])

    return jsonify(clusters)

if __name__ == "__main__":
    app.run(debug=True)
