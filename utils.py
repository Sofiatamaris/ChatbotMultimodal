import json # Importa la biblioteca `json` para trabajar con datos en formato JSON
from langchain.schema.messages import HumanMessage, AIMessage # Importa clases de mensaje del esquema `langchain`
from datetime import datetime# Importa la clase `datetime` para manejar fechas y horas
import yaml# Importa la biblioteca `yaml` para trabajar con archivos YAML de configuración

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    
    #Guarda el historial de chat en un archivo JSON
def save_chat_history_json(chat_history, file_path):
    with open(file_path, "w") as f:
        json_data = [message.dict() for message in chat_history]
        json.dump(json_data, f)

    #Carga el historial de chat desde un archivo JSON.
def load_chat_history_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [HumanMessage(**message) if message["type"] == "human" else AIMessage(**message) for message in json_data]
        return messages

    #Obtiene la fecha y hora actual en formato legible
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #Obtiene la ruta a la imagen de avatar según el tipo de remitente
def get_avatar(sender_type):
    if sender_type == "human":
        return "chat_icons/user_image.png"
    else:
       return "chat_icons/bot_image.png"