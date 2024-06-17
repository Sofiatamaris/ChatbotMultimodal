
import streamlit as st#Importa la librería Streamlit para crear aplicaciones web.
from llm_chains import load_normal_chain, load_pdf_chat_chain#Importa funciones para cargar diferentes cadenas de LLM (Large Language Model)
from streamlit_mic_recorder import mic_recorder#Importa una función para grabar audio de entrada usando el micrófono.
from utils import get_timestamp, load_config, get_avatar#Importa funciones auxiliares para generar marcas de tiempo, cargar configuraciones y obtener avatares.
from image_handler import handle_image#Importa una función para procesar imágenes subidas.
from audio_handler import transcribe_audio#Importa una función para transcribir archivos de audio.
from html_templates import css#Importa estilos CSS para formatear la aplicación web.
from database_operations import load_last_k_text_messages, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history#Importa funciones para interactuar con la base de datos del historial de chat (cargar mensajes pasados, guardar nuevos mensajes, etc.).
import sqlite3#Importa la librería SQLite para operaciones con la base de datos.


config = load_config() #Carga los ajustes de configuración desde config 

#Guarda en caché el resultado de la función para mejorar el rendimiento.
@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        print("")
        return load_pdf_chat_chain()
    return load_normal_chain()

#Comprueba si se está iniciando una nueva sesión.
def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key


#Elimina el historial de chat de la sesión actual de la base de datos.
def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

#Limpia el cache del historial de chats
def clear_cache():
    st.cache_resource.clear()

def main():
    st.title("✨Chatbot")
    st.write(css, unsafe_allow_html=True)
    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    #Barra lateral de selección de sesiones de chat.
    st.sidebar.title("Historial de Chat")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Selecciona una sesion", chat_sessions, key="session_key", index=index)
    pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
    pdf_toggle_col.toggle(" ", key="pdf_chat", value=False, on_change=clear_cache)
    with voice_rec_col:
        voice_recording=mic_recorder(start_prompt="Grabar Audio",stop_prompt="Parar Grabacion", just_once=True)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Eliminar sesion", on_click=delete_chat_session_history)
    clear_cache_col.button("Limpiar cache", on_click=clear_cache)
    
    #Contenedor para mostrar el historial de chat y la entrada de usuario.
    chat_container = st.container()
    user_input = st.chat_input("Escribe tu mensaje aqui", key="user_input")
    
    
    uploaded_audio = st.sidebar.file_uploader("Subir un audio", type=["wav", "mp3", "ogg"], key=st.session_state.audio_uploader_key)
    uploaded_image = st.sidebar.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])
    

    
    #Cargador de archivos para subir audio en formatos WAV, MP3 u OGG, almacenando en archivos
    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input = "Summarize this text: " + transcribed_audio, chat_history=[])
        save_audio_message(get_session_key(), "human", uploaded_audio.getvalue())
        save_text_message(get_session_key(), "ai", llm_answer)
        st.session_state.audio_uploader_key += 2

    #Si hay una grabación de voz transcribe la grabación de voz 
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input = transcribed_audio, 
                                   chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_audio_message(get_session_key(), "human", voice_recording["bytes"])
        save_text_message(get_session_key(), "ai", llm_answer)

    #Procesa la imagen, combinando la imagen con el mensaje de texto, y almacena la respuesta
    if user_input:
        if uploaded_image:
            with st.spinner("Procesando imagen..."):
                llm_answer = handle_image(uploaded_image.getvalue(), user_input)
                save_text_message(get_session_key(), "human", user_input)
                save_image_message(get_session_key(), "human", uploaded_image.getvalue())
                save_text_message(get_session_key(), "ai", llm_answer)
                user_input = None


        if user_input:
            llm_chain = load_chain()
            llm_answer = llm_chain.run(user_input = user_input, 
                                       chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
            save_text_message(get_session_key(), "human", user_input)
            save_text_message(get_session_key(), "ai", llm_answer)
            user_input = None

    #Si la sesión actual no es una nueva sesión o si se ha seleccionado una nueva sesión, Crea un bloque para mostrar el historial de chat,Carga el historial de chat de la sesión actual.
    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")

        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()
