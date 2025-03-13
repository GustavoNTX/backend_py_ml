API de Machine Learning para Classificação de Gamers
Este é o backend de uma aplicação de Machine Learning desenvolvida em Python com Flask, que classifica gamers com base em seus atributos utilizando aprendizado supervisionado e não supervisionado.

A API processa dados sobre habilidades dos jogadores e os agrupa em categorias usando K-Means (aprendizado não supervisionado) e prevê a categoria de novos jogadores usando um modelo supervisionado.

🚀 Funcionalidades
✔️ Classificação de Gamers: Agrupa jogadores automaticamente em 5 categorias com base em seu estilo de jogo.
✔️ Previsão de Categoria: Utiliza um modelo de aprendizado supervisionado para prever a categoria de um novo jogador com base em seus atributos.
✔️ Análise de Clusters: Retorna os grupos de jogadores formados, exibindo os dados de cada um.
✔️ Integração com o Frontend: Responde às requisições do frontend, fornecendo dados para exibição em gráficos interativos.

🛠️ Tecnologias Utilizadas
Python 🐍
Flask (Framework Web)
Flask-CORS (Para permitir requisições do frontend)
Scikit-Learn (Machine Learning)
NumPy e JSON (Manipulação de dados)
📡 Rotas da API
🔹 Verificar Status
🔗 GET /
Retorno: { "message": "API de Classificação de Gamers está online!" }

🔹 Previsão de Categoria (Supervisionado)
🔗 POST /prever
Envia:
{
  "habilidade": 80,
  "tempo_jogo": 200,
  "familiaridade": 90,
  "competitividade": 70,
  "imersao": 85
}
Recebe:
{ "categoria_prevista": "Tryhard Gamer" }
🔹 Obter Clusters de Gamers (Não Supervisionado)
🔗 GET /clusters
Retorno:
{
  "0": {
    "centro": [72.4, 180.3, 88.2, 65.1, 80.7],
    "jogadores": [
      { "id": 1, "habilidade": 75, "categoria": "Tryhard Gamer" },
      { "id": 2, "habilidade": 80, "categoria": "Truegamer" }
    ]
  }
}
