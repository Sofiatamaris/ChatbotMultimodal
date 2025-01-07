
Descripción general

Este proyecto trata de integrar diferentes modelos de IA para manejar audio, imágenes  en una única interfaz de chat creada con streamlit. 

Se uso Whisper AI para audio, LLaVA para procesamiento de imágenes, Qlite para la base de datos del historial de chat y Chroma DB para archivos PDF (Este ultimo no se termino de desarrollar)


Para la instalacion del chat con IA, clonar el repositorio y seguir estos pasos:

1. Crear un entorno virtual actualmente se esta usando Python 3.12.3

2. Actualizar pip ```pip install --update pip```

3. Requisitos de instalación: ```pip install -r requisitos.txt```

4. Configuración de modelos locales. descargue los modelos que desea implementar.(https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main) está el modelo de llava para el chat de imágenes (ggml-model-q5_k.gguf y mmproj-model-f16. guf).
Y el [modelo mistral cuantificado] (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) forma TheBloke (mistral-7b-instruct-v0.1.Q5_K_M.gguf).

5. Personalizar archivo de configuración: verifique el archivo de configuración y cámbielo según los modelos que descargó.

6. Ingrese comandos en la terminal:
    1. ```python3 data_operaciones.py``` Esto inicializará la base de datos sqlite para las sesiones de chat.
    2. ```streamlit run app.py```

