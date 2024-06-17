import torch # Importa la biblioteca Torch para el procesamiento de tensores y redes neuronales
from transformers import pipeline # Importa la función `pipeline` de la biblioteca Transformers para crear pipelines de procesamiento de lenguaje natural
import librosa # Importa la biblioteca Librosa para el análisis y procesamiento de audio
import io # Importa el módulo `io` para operaciones de entrada y salida de datos
from utils import load_config # Importa la función `load_config` desde el módulo `utils` para cargar la configuración de la aplicación
config = load_config()

#Convierte datos de audio en formato binario (bytes) a un array NumPy.
def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)#Crea un objeto de archivo en memoria a partir de los bytes de audio
    audio, sample_rate = librosa.load(audio_bytes)# Carga el audio y obtiene la frecuencia de muestreo
    print(sample_rate)# Imprime la frecuencia de muestreo (se puede eliminar en un entorno de producción)
    return audio # Devuelve el audio como un array NumPy

# Transcribe un archivo de audio a texto utilizando un modelo de reconocimiento de voz.
def transcribe_audio(audio_bytes):
    device = "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",  # Crea un pipeline para el reconocimiento automático de voz
        model=config["whisper_model"],# Especifica el modelo de reconocimiento de voz
        chunk_length_s=30,## Define la longitud de los segmentos de audio para procesar (30 segundos en este caso)
        device=device, ## Establece el dispositivo de computación
    )   

    audio_array = convert_bytes_to_array(audio_bytes) # Convierte los bytes de audio a un array NumPy
    prediction = pipe(audio_array, batch_size=1)["text"] # Realiza la transcripción del audio

    return prediction # Devuelve el texto transcrito
