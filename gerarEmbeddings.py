import google.generativeai as generativeai
import pandas as pd
from dotenv import load_dotenv
import os
import time
from google.api_core import exceptions

# Carrega as variáveis de ambiente e configura a API
load_dotenv()
chave_secreta = os.getenv('API_KEY')
print("API Key carregada.")
generativeai.configure(api_key=chave_secreta)

# Lê os dados da planilha do Google
csv_url = 'https://docs.google.com/spreadsheets/d/1pr9QbEENce-9NFrTtLdrfUd_swzQoOs0FDbShaqMECQ/export?format=csv&id=1pr9QbEENce-9NFrTtLdrfUd_swzQoOs0FDbShaqMECQ'
df = pd.read_csv(csv_url)
print("Dados carregados com sucesso:")
print(df.head())

# Define o modelo de embedding
model = 'models/gemini-embedding-exp-03-07'

# --- INÍCIO DA LÓGICA FINAL COM NOVA TENTATIVA AUTOMÁTICA ---

print("\nIniciando a geração dos embeddings com nova tentativa automática...")

# 1. Prepara a lista de todos os textos
to_embed = (df["Titulo"] + "\n" + df["Conteúdo"]).tolist()

# 2. Define o tamanho de cada "pedaço"
CHUNK_SIZE = 5
embeddings_list = []

for i in range(0, len(to_embed), CHUNK_SIZE):
    # Pega o próximo pedaço da lista
    chunk = to_embed[i:i + CHUNK_SIZE]
    
    print(f"Processando pedaço {i//CHUNK_SIZE + 1}...")
    
    # Loop de nova tentativa para o pedaço atual
    while True:
        try:
            # Tenta fazer a chamada para a API
            result = generativeai.embed_content(model=model,
                                                 content=chunk,
                                                 task_type="retrieval_document")
            
            # Se bem-sucedido, adiciona os embeddings e sai do loop de tentativa
            embeddings_list.extend(result['embedding'])
            print("... Sucesso!")
            break  # Sai do loop 'while' e vai para o próximo pedaço no 'for'

        except exceptions.ResourceExhausted as e:
            # Se o erro de limite for capturado, espera 60 segundos e tenta de novo
            print("... Limite da API atingido. Aguardando 60 segundos para tentar novamente.")
            time.sleep(60)
            
# 3. Adiciona a lista completa de embeddings ao DataFrame
df["Embeddings"] = embeddings_list

# --- FIM DA LÓGICA FINAL ---

print("\nTodos os embeddings foram gerados com sucesso!")

# Salva o DataFrame com os embeddings em um arquivo pickle
import pickle
pickle.dump(df, open('datasetEmbedding2025.pkl', 'wb'))
print("Arquivo 'datasetEmbedding2025.pkl' salvo.")

# Carrega e imprime o arquivo para verificação
modeloEmbeddings = pickle.load(open('datasetEmbedding2025.pkl', 'rb'))
print("\nVerificação do arquivo salvo:")
print(modeloEmbeddings)