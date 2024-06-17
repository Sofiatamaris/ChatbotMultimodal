# Importa bibliotecas para trabajar con LLM chains y procesamiento de texto
from prompt_templates import memory_prompt_template, pdf_chat_prompt
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from operator import itemgetter
from utils import load_config
import chromadb

# Carga la configuración de la aplicación desde un archivo YAML
config = load_config()

# Función para cargar el modelo Ollama
def load_ollama_model():
    llm = Ollama(model=config["ollama_model"])
    return llm

# Función para crear un modelo LLM
def create_llm(model_path = config["ctransformers"]["model_path"]["large"], model_type = config["ctransformers"]["model_type"], model_config = config["ctransformers"]["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

# Función para crear embeddings de palabras
def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

# Función para crear la memoria de la conversación
def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

# Función para crear un prompt a partir de una plantilla
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

# Función para crear una cadena LLM
def create_llm_chain(llm, chat_prompt):
    return LLMChain(llm=llm, prompt=chat_prompt)
    
# Función para cargar una cadena de chat normal
def load_normal_chain():
    return chatChain()

# Función para cargar la base de datos vectorial
def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma

# Función para cargar la cadena de chat relacionada con PDF
def load_pdf_chat_chain():
    return pdfChatChain()

# Función para cargar una cadena de recuperación y respuesta
def load_retrieval_chain(llm, vector_db):
    return RetrievalQA.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}), verbose=True)

# Función para crear un runnable para la cadena de chat PDF
def create_pdf_chat_runnable(llm, vector_db, prompt):
    runnable = (
        {
        "context": itemgetter("human_input") | vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}),
        "human_input": itemgetter("human_input"),
        "history" : itemgetter("history"),
        }
    | prompt | llm.bind(stop=["Human:"]) 
    )
    return runnable

# Clase para la cadena de chat PDF
class pdfChatChain:

    def __init__(self):
        vector_db = load_vectordb(create_embeddings())
        llm = create_llm()
        #llm = load_ollama_model()
        prompt = create_prompt_from_template(pdf_chat_prompt)
        self.llm_chain = create_pdf_chat_runnable(llm, vector_db, prompt)

    def run(self, user_input, chat_history):
        print("Pdf chat chain is running...")
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history})

# Clase para la cadena de chat general
class chatChain:

    def __init__(self):
        llm = create_llm()
        #llm = load_ollama_model()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt)

    def run(self, user_input, chat_history):
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history} ,stop=["Human:"])["text"]