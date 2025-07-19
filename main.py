from groq import Groq
import whisper
import os
from datetime import datetime

# Configurações atualizadas
GROQ_API_KEY = "your API key"  # ← Substitua pela sua chave!
MODELO_WHISPER = "base"
MODELO_GROQ = "llama3-70b-8192"  # Modelo atualizado (substitui o Mixtral)

def transcrever_video(caminho_video):
    """Transcreve o vídeo usando Whisper"""
    try:
        print(f"🔈 Processando: {os.path.basename(caminho_video)}...")
        modelo = whisper.load_model(MODELO_WHISPER)
        resultado = modelo.transcribe(
            caminho_video,
            fp16=False,
            language='pt',
            verbose=True  # Mostra progresso
        )
        
        nome_arquivo = f"transcricao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            f.write(f"Fonte: {caminho_video}\n")
            f.write(f"Modelo: {MODELO_WHISPER}\n")
            f.write(f"Data: {datetime.now()}\n\n")
            f.write(resultado["text"])
        
        print(f"✅ Transcrição salva em '{nome_arquivo}'")
        return resultado["text"]
    
    except Exception as e:
        print(f"❌ Falha na transcrição: {str(e)}")
        return None

def analisar_com_groq(texto):
    """Analisa o texto com Groq usando modelo atual"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        print("🧠 Gerando análise com Groq...")
        resposta = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Você é um especialista em resumir conteúdo. Responda em português."
                },
                {
                    "role": "user",
                    "content": f"Resuma em 5 tópicos principais:\n\n{texto[:10000]}"  # Limite de tokens
                }
            ],
            model=MODELO_GROQ,
            temperature=0.3,
            max_tokens=1024
        )
        
        return resposta.choices[0].message.content
    
    except Exception as e:
        print(f"❌ Erro na API Groq: {str(e)}")
        print("\nDicas:")
        print("- Verifique sua API key em https://console.groq.com/keys")
        print("- Consulte os modelos disponíveis em https://console.groq.com/docs/models")
        return None

def main():
    caminho_video = os.path.join("videos", "video1.mp4")
    
    if not os.path.exists(caminho_video):
        print(f"Arquivo não encontrado:\n{os.path.abspath(caminho_video)}")
        return
    
    texto = transcrever_video(caminho_video)
    if not texto:
        return
    
    analise = analisar_com_groq(texto)
    if analise:
        print("\n📝 Análise Gerada:")
        print(analise)
        
        with open("analise.txt", "w", encoding="utf-8") as f:
            f.write(analise)
        print("\n✅ Análise salva em 'analise.txt'")

if __name__ == "__main__":
    main()