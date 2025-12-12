from minio import Minio
from minio.error import S3Error
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

import threading
import functools
import time


class MinioEngine: 
    def __init__(self, host, access_key, secret_key, secure):
        self.host = host
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure

    def object_engine(self): 
        return Minio(
            self.host,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )


class RagUtils: 
    @staticmethod
    def _get_minio_client(minio_engine):
        """Retourne toujours un client MinIO valide."""
        if isinstance(minio_engine, Minio):
            return minio_engine
        return minio_engine.object_engine()

    @staticmethod
    def download_ressources(minio_engine, bucket_name: str, local_dir="./pdfs"):
        """
        Fonction de telechargement des ressources depuis le bucket minio.
        Accepte MinioEngine OU Minio directement.
        """

        client = RagUtils._get_minio_client(minio_engine)

        os.makedirs(local_dir, exist_ok=True)

        try:
            objects = client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                if obj.object_name.lower().endswith(".pdf"):
                    local_path = os.path.join(local_dir, obj.object_name)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    client.fget_object(bucket_name, obj.object_name, local_path)
                    print(f"Téléchargé : {obj.object_name}")
        except S3Error as e:
            print(f"Erreur MinIO : {e}")

    @staticmethod
    def create_base_of_knowledge(minio_engine, bucket_name: str, nb_chuncks: int, 
                                 val_overlap: int = 0, model_embedding: str = "embeddinggemma"):
        """
        Initialise la base de connaissance :
        - Téléchargement depuis MinIO
        - Chunking
        - Embedding
        - FAISS
        """

        print("🔍 Vérification du contenu du bucket...")

        client = RagUtils._get_minio_client(minio_engine)

        objects = list(client.list_objects(bucket_name, recursive=True))

        if len(objects) == 0:
            print("⚠️  Bucket vide. Impossible de créer la base de connaissance.")
            return

        print("📥 Téléchargement des ressources depuis MinIO...")

        RagUtils.download_ressources(
            minio_engine=minio_engine,
            bucket_name=bucket_name,
            local_dir="./pdfs"
        )

        print("📁 Téléchargement terminé.")

        loader = DirectoryLoader(
            "./pdfs",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=nb_chuncks,
            chunk_overlap=val_overlap,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model=model_embedding)

        vector_store = FAISS.from_documents(chunks, embeddings)

        vector_store.save_local("faiss_index")

        print("📚 Base de connaissance initialisée.")
        print(f"📄 {len(objects)} fichiers trouvés dans le bucket.")

    @staticmethod
    def load_base_of_knowledge(embeddings_model: str = "embeddinggemma", load_path="faiss_index"):
        embeddings = OllamaEmbeddings(model=embeddings_model)
        base_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        return base_store

    @staticmethod
    def preprocessing_text(text):
        pass

    @staticmethod
    def preprocessing_pdf(pdfile):
        pass



def timer(max_minutes):
    max_seconds = max_minutes * 60

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]

            def target():
                result[0] = func(*args, **kwargs)

            # On exécute la fonction dans un thread
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=max_seconds)

            # Si le thread est toujours en vie → timeout
            if thread.is_alive():
                return None

            return result[0]

        return wrapper
    return decorator
