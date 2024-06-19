from flask import Flask, request, jsonify
from flasgger import Swagger
import pickle
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
import os 
from flask_sqlalchemy import SQLAlchemy  # Importa o SQLAlchemy
from threading import Thread

app = Flask(__name__)

# Configuração do banco de dados
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'  # Usando SQLite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuração do Swagger
swagger = Swagger(app)

from model import preprocess_text, prepare_fasttext_vectors, predict_xgboost

# Carregar os modelos pré-treinados
with open('./models/xgboost_negative_vs_rest.pkl', 'rb') as file:
    model_neg_vs_rest = pickle.load(file)

with open('./models/xgboost_positive_vs_neutral.pkl', 'rb') as file:
    model_pos_vs_neutral = pickle.load(file)

# Definir a variável de ambiente para o token do Slack
os.environ["SLACK_BOT_TOKEN"] = 'xoxp-7223665090662-7230184266179-7253141460016-5c83eadf05f7eb4a917af42aa7f999a6'

# Inicializar o aplicativo Slack
slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))
handler = SlackRequestHandler(slack_app)

def send_slack_alert(message):
    try:
        response = slack_app.client.chat_postMessage(
            channel="#sentiment_tracker_inteli",
            text=message
        )
        app.logger.info(f"Message sent to Slack channel: {response}")
    except Exception as e:
        app.logger.error(f"Error sending message to Slack: {e}")
        
# Definição do modelo para armazenar as frases
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    preprocessed_text = db.Column(db.String(500))
    sentiment = db.Column(db.String(50))

    def __init__(self, text, preprocessed_text, sentiment):
        self.text = text
        self.preprocessed_text = preprocessed_text
        self.sentiment = sentiment

# Criar o banco de dados e as tabelas
with app.app_context():
    db.create_all()

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """
    Endpoint para pré-processar um texto.
    ---
    parameters:
      - name: text
        in: body
        type: string
        required: true
        description: Texto a ser pré-processado.
    responses:
      200:
        description: Texto pré-processado com sucesso.
        schema:
          type: object
          properties:
            preprocessed_text:
              type: string
              description: Texto pré-processado.
    """
    data = request.get_json()
    text = data['text']
    preprocessed_text = preprocess_text(text)
    return jsonify({'preprocessed_text': preprocessed_text})
  
@app.route('/fasttext', methods=['POST'])
def fasttext():
    """
    Endpoint para obter o vetor de palavras de um texto.
    ---
    parameters:
      - name: text
        in: body
        type: string
        required: true
        description: Texto para o qual o vetor de palavras será calculado.
      - name: embedding_file
        in: body
        type: string
        required: true
        description: Caminho do arquivo de embedding.
    responses:
      200:
        description: Vetor de palavras calculado com sucesso.
        schema:
          type: object
          properties:
            word_vector:
              type: array
              description: Vetor de palavras.
              items:
                type: number
    """
    data = request.get_json()
    text = data['text']
    fasttext_model_path = './data/cc.en.100.vec'
    vector_df = prepare_fasttext_vectors(text, fasttext_model_path)
    word_vector = vector_df.tolist()
    return jsonify({'word_vector': word_vector})
  
@app.route("/analise_xgboost", methods=["POST"])
def analyze_sentiment():
    """
    Endpoint para análise de sentimento de um texto.
    ---
    parameters:
      - name: text
        in: body
        type: string
        required: true
        description: Texto a ser analisado.
    responses:
      200:
        description: Sentimento analisado com sucesso.
        schema:
          type: object
          properties:
            sentiment:
              type: string
              description: Sentimento do texto.
    """
    data = request.get_json()
    text = data['text']
    preprocessed_text = preprocess_text(text)
    fasttext_model_path = './data/cc.en.100.vec'
    
    # Certifique-se de passar os modelos já carregados
    prediction = predict_xgboost(text, fasttext_model_path, model_neg_vs_rest, model_pos_vs_neutral)

    if prediction == 1:
        sentiment_message = "positive"
    elif prediction == -1:
        sentiment_message = "negative"
    else:
        sentiment_message = "neutral"

    new_comment = Comment(text=text, preprocessed_text=preprocessed_text, sentiment=sentiment_message)
    db.session.add(new_comment)
    db.session.commit()

    
    send_slack_alert(f"Analyzed text: {text}\nPrediction: {sentiment_message}")
    
    return jsonify({'prediction': sentiment_message})
  
@app.route('/test_slack_message', methods=['GET'])
def test_slack_message():
    try:
        send_slack_alert("Test message from Flask app")
        return jsonify({"message": "Test message sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      
def run_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = Thread(target=run_flask)

    flask_thread.start()

    flask_thread.join()
