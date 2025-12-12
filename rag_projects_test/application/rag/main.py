import os
import threading
import functools
from datetime import datetime
from rag.rag_utils import RagUtils as r
from rag.rag_utils import MinioEngine as me
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import ollama
from dotenv import load_dotenv

load_dotenv("../.env")
LLMLOG_FILE = "llmlog.txt"
BUCKET = "documents-edu"


# ------------------- TIMER -------------------
def timer(max_minutes):
    max_seconds = max_minutes * 60

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=max_seconds)

            if thread.is_alive():
                return None  # Timeout
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


# ------------------- RAG CHAT -------------------
class RagChat:
    def __init__(self, minio_engine: me):
        self.minio_engine = minio_engine

    @timer(max_minutes=1)
    def response_ollama(self, prompt, similarity: int = 5, llm: str = "mistral:latest"):
        if 'faiss_index' not in os.listdir(os.getcwd()):
            r.create_base_of_knowledge(minio_engine=self.minio_engine, bucket_name=BUCKET, nb_chuncks=1000)

        vector_base = r.load_base_of_knowledge()
        relevant_docs = vector_base.similarity_search(prompt, k=similarity)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt_ = f"""
        Tu es un assistant éducatif.
        CONTEXTE:
        {context}

        QUESTION:
        {prompt}

        Réponse :
        """

        response = ollama.chat(
            model=llm,
            messages=[{"role": "user", "content": prompt_}]
        )

        return {"message": response["message"]["content"], "metadata": {"model": llm}}

    @timer(max_minutes=2)
    def response_deepseek(self, prompt, similarity: int = 5):
        if 'faiss_index' not in os.listdir(os.getcwd()):
            r.create_base_of_knowledge(minio_engine=self.minio_engine)

        vector_base = r.load_base_of_knowledge()
        relevant_docs = vector_base.similarity_search(prompt, k=similarity)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        endpoint = "https://models.github.ai/inference"
        # model = "deepseek/DeepSeek-V3-0324"
        model = "meta/Llama-4-Scout-17B-16E-Instruct"
        token = os.getenv("TOKEN_MICROSOFT_IA_DEEP_SEEK")

        prompt_ = f"""
        Tu es un assistant éducatif.
        Utilise UNIQUEMENT les informations suivantes pour répondre.
        Si ce n'est pas dans le contexte, dis que ce n'est pas disponible.

        CONTEXTE :
        {context}

        QUESTION :
        {prompt}

        Réponse :
        """

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

        answer = client.complete(
            messages=[
                SystemMessage("Tu es un assistant RAG spécialisé en éducation."),
                UserMessage(prompt_),
            ],
            temperature=0.8,
            top_p=0.1,
            max_tokens=2048,
            model=model
        )

        return {"message": answer.choices[0].message.content, "metadata": answer}

    # ------------------- LOG -------------------
    def raglog(self, prompt, model_name, response, metadata="metadata"):
        file_exists = os.path.isfile(LLMLOG_FILE)
        headers = "date,model,prompt,response,metadata\n"

        with open(LLMLOG_FILE, "a") as f:
            if not file_exists:
                f.write(headers)
            date = datetime.now()
            log = f"{date},>>{model_name}>>,{prompt},{response},{metadata}\n"
            f.write(log)

    # ------------------- CHAT -------------------
    def chat(self, prompt):
        answer = None
        try:
            print(">>>> Deepseek Response >>>")
            answer = self.response_deepseek(prompt)
            if answer is None:
                raise TimeoutError("DeepSeek timeout")
            model_name = answer.get("metadata")[0].get("model") if isinstance(answer.get("metadata"), list) else "DeepSeek"
            self.raglog(prompt, model_name=model_name, response=answer.get("message"))
            return answer

        except Exception as e:
            print(f"Erreur avec DeepSeek : {e}. Passage à Ollama.")
            try:
                answer = self.response_ollama(prompt)
                if answer is None:
                    raise TimeoutError("Ollama timeout")
                self.raglog(prompt, model_name="mistral:latest -- ollama", response=answer.get("message"))
                return answer

            except Exception as e2:
                print(f"Erreur avec Ollama : {e2}. Envoi du message par défaut.")
                default_response = "Crédits API DeepSeek terminés et Ollama a mis trop de temps à répondre."
                self.raglog(prompt, model_name="Deepseek & Ollama timeout", response=default_response)
                return default_response
