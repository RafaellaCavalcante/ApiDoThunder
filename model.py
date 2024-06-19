import re
import emoji
import os
import numpy as np
import pandas as pd
import pickle
import fasttext
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer

# Função para converter emojis em palavras
def convert_emojis(text):
    """
    Converte emojis no texto em seus nomes correspondentes.

    Args:
    text (str): A string de entrada que pode conter emojis.

    Returns:
    str: A string com emojis convertidos em palavras.
    """
    return emoji.demojize(text)

# Função para limpar o texto e substituir apóstrofos por espaços
def clean_text(text):
    """
    Limpa o texto removendo URLs, tags HTML, caracteres especiais e números. 
    Também substitui apóstrofos por espaços e corrige espaços extras.

    Args:
    text (str): A string de entrada que será limpa.

    Returns:
    str: A string limpa sem URLs, tags HTML e caracteres especiais.
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)  # Remove tags HTML
    text = re.sub(r'\'', ' ', text)  # Substitui apóstrofos por espaços
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove caracteres especiais e números, exceto espaços
    text = re.sub(r'\s+', ' ', text).strip()  # Corrige espaços extras
    return text

# Função para tokenizar o texto e converter para minúsculas
def tokenize(text):
    """
    Tokeniza o texto, dividindo-o em palavras e convertendo para minúsculas.

    Args:
    text (str): A string de entrada a ser tokenizada.

    Returns:
    list: Lista de tokens (palavras) em minúsculas.
    """
    tokens = word_tokenize(text.lower())
    return tokens

# Função para remover stopwords
def remove_stopwords(tokens):
    """
    Remove stopwords (palavras irrelevantes) da lista de tokens.

    Args:
    tokens (list): Lista de tokens das quais as stopwords serão removidas.

    Returns:
    list: Lista de tokens filtrados sem as stopwords.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# Função para aplicar lematização
def apply_lemmatization(tokens):
    """
    Aplica lematização aos tokens para reduzir as palavras à sua forma base.

    Args:
    tokens (list): Lista de tokens a serem lematizados.

    Returns:
    list: Lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Lista de palavras a serem mantidas
words_to_keep = ['uber']

# Função para corrigir a ortografia, mantendo palavras específicas
def correct_spelling(text):
    """
    Corrige a ortografia do texto, preservando palavras específicas que não 
    devem ser alteradas.

    Args:
    text (str): A string de entrada a ser corrigida ortograficamente.

    Returns:
    str: A string com a ortografia corrigida.
    """
    for i, word in enumerate(words_to_keep):
        text = text.replace(word, f'PLACEHOLDER_{i}')
    corrected_text = str(TextBlob(text).correct())
    for i, word in enumerate(words_to_keep):
        corrected_text = corrected_text.replace(f'PLACEHOLDER_{i}', word)
    return corrected_text

# Função principal de pré-processamento para uma única frase
def preprocess_text(text):
    """
    Executa uma série de etapas de pré-processamento no texto: 
    conversão para minúsculas, conversão de emojis para palavras, 
    limpeza do texto, correção ortográfica, tokenização, remoção de 
    stopwords e lematização.

    Args:
    text (str): A string de entrada a ser pré-processada.

    Returns:
    str: A string processada após todas as etapas de pré-processamento.
    """
    text = text.lower()  # Converte para minúsculas
    text = convert_emojis(text)
    text = clean_text(text)
    text = correct_spelling(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    lemmatized_tokens = apply_lemmatization(tokens)
    return ' '.join(lemmatized_tokens)  # Retorna a frase lematizada

def prepare_fasttext_vectors(sentence, vec_path):
    """
    Carrega os embeddings do fastText a partir de um arquivo .vec (cc.en.100.vec) e converte uma frase em um vetor de embeddings.

    Args:
    sentence (str): Frase de entrada.
    vec_path (str): Caminho para o arquivo .vec com os embeddings.

    Returns:
    np.array: Vetor que representa a frase.
    """
    # Verifica se o arquivo .vec existe
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Arquivo de embeddings não encontrado no caminho: {vec_path}")
    
    # Carrega os embeddings do arquivo .vec
    embeddings = {}
    with open(vec_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    
    # Converte a frase em um vetor usando a média dos embeddings das palavras
    words = sentence.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    
    if len(word_vectors) == 0:
        # Retorna um vetor de zeros se nenhuma palavra da frase estiver nos embeddings
        return np.zeros(len(next(iter(embeddings.values()))))
    
    # Calcula a média dos vetores das palavras
    return np.mean(word_vectors, axis=0)

def classify_text_neg_rest(vectorized_text, model):
    """
    Classifica uma frase como negativa (0) ou não negativa (1) usando um modelo XGBoost treinado.

    Inputs:
        vectorized_text (np.ndarray): Vetor representando a frase vetorizada.
        model (XGBClassifier): Modelo XGBoost treinado.

    Output: int: Classificação da frase (0: negativa, 1: não negativa).
    """
    # Verificar se o vetor é unidimensional e transformar em um array 2D se necessário
    if len(vectorized_text.shape) == 1:
        vectorized_text = vectorized_text.reshape(1, -1)

    # Fazer a predição usando o modelo carregado
    prediction = model.predict(vectorized_text)
    
    return int(prediction[0])

def classify_text_pos_neutral(vectorized_text, model):
    """
    Classifica uma frase como positiva (2) ou neutra (1) usando um modelo XGBoost treinado.

    Inputs:
        vectorized_text (np.ndarray): Vetor representando a frase vetorizada.
        model (XGBClassifier): Modelo XGBoost treinado.

    Output: int: Classificação da frase (2: positiva, 1: neutra).
    """
    # Verificar se o vetor é unidimensional e transformar em um array 2D se necessário
    if len(vectorized_text.shape) == 1:
        vectorized_text = vectorized_text.reshape(1, -1)

    # Fazer a predição usando o modelo carregado
    prediction = model.predict(vectorized_text)
    
    return int(prediction[0])

def predict_xgboost(text, fasttext_model_path, model_neg_vs_rest, model_pos_vs_neutral):
    """
    Pré-processa e classifica um texto dado usando modelos de FastText e classificadores pré-treinados.

    Esta função realiza o pré-processamento do texto, vetorização usando um modelo FastText, e 
    classifica o texto em uma das três categorias: negativo, neutro ou positivo. 
    Primeiramente, classifica se o texto é negativo ou não, e se não for negativo, 
    classifica entre positivo e neutro.

    Args:
    text (str): A string de entrada contendo o texto a ser classificado.
    fasttext_model_path (str): O caminho para o modelo FastText pré-treinado para vetorização.
    model_neg_vs_rest (XGBClassifier): Modelo treinado para classificar entre negativo e não negativo.
    model_pos_vs_neutral (XGBClassifier): Modelo treinado para classificar entre positivo e neutro.

    Returns:
    int: A categoria do texto classificado.
         -1 indica que o texto é negativo.
         0 indica que o texto é neutro.
         1 indica que o texto é positivo.
    """
    # Pré-processar o texto
    preprocessed_text = preprocess_text(text)
    
    # Vetorizar o texto
    vectorized_text = prepare_fasttext_vectors(preprocessed_text, fasttext_model_path)
    
    # Classificar o texto como negativo vs resto
    neg_vs_rest = classify_text_neg_rest(vectorized_text, model_neg_vs_rest)
    
    if neg_vs_rest == 0:  # Se não for negativo, classificar entre positivo e neutro
        pos_vs_neutral = classify_text_pos_neutral(vectorized_text, model_pos_vs_neutral)
        return 1 if pos_vs_neutral == 1 else 0  # 1: positivo, 0: neutro
    else:
        return -1  # -1: negativo