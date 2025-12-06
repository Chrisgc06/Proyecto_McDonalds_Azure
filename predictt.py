import streamlit as st
import requests
import os 
import csv 
from datetime import datetime
from io import BytesIO
from PIL import Image 
import pandas as pd

BASE_ENDPOINT_URL = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/c8c26e88-95cb-485b-9018-19be9111d171/detect/iterations/"
PREDICTION_KEY = "3pXtAr2NT2A69563TCPpfDHZPlIwCaIsDGbeSZNj1jYGzdf7Kg2sJQQJ99BLACYeBjFXJ3w3AAAIACOG3sh0"
CONFIDENCE_THRESHOLD = 0.50

AVAILABLE_ITERATIONS = [
    "Iteration9 (Recomendada)", 
    "Iteration8", 
    "Iteration7", 
    "Iteration6",
    "Iteration5"
]

HEADERS = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

CSV_FILE = 'predicciones_log.csv'

def log_prediction(data):
    headers = [
        'timestamp', 'image_source', 'tag_name', 'probability', 
        'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height'
    ]
    file_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='\n', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def clasificar_imagen(image_data, file_name, endpoint_url, iteration_name):
    status_msg = st.info(f"Enviando imagen a Azure Custom Vision (Modelo: {iteration_name})...")

    try:
        response = requests.post(endpoint_url, headers=HEADERS, data=image_data, timeout=30)
        response.raise_for_status() 
        status_msg.empty()
    except requests.exceptions.RequestException as e:
        status_msg.error(f"Error en la conexión a Azure. Verifica el ENDPOINT y la clave. Error: {e}")
        return None

    resultado = response.json()
    
    if 'predictions' not in resultado or not resultado['predictions']:
        st.warning("La API de Azure no retornó ninguna predicción para esta imagen.")
        return None

    st.success(f"Detección Exitosa: Resultados de {iteration_name}")
    
    data_for_df = []
    timestamp = datetime.now().isoformat()
    low_confidence_detected = False

    for pred in resultado["predictions"]:
        probability = pred['probability']
        tag_name = pred['tagName']
        bbox = pred['boundingBox']
        
        if probability < CONFIDENCE_THRESHOLD:
             low_confidence_detected = True
        
        if probability >= CONFIDENCE_THRESHOLD:
            record = {
                'timestamp': timestamp, 'image_source': file_name, 'tag_name': tag_name,
                'probability': f"{probability:.4f}", 'bbox_left': f"{bbox['left']:.4f}", 
                'bbox_top': f"{bbox['top']:.4f}", 'bbox_width': f"{bbox['width']:.4f}", 
                'bbox_height': f"{bbox['height']:.4f}"
            }
            log_prediction(record)

        data_for_df.append({
            'Etiqueta': tag_name,
            'Confianza (%)': probability * 100,
            'Bbox (Left)': f"{bbox['left']:.4f}", 'Bbox (Top)': f"{bbox['top']:.4f}",
            'Bbox (Width)': f"{bbox['width']:.4f}", 'Bbox (Height)': f"{bbox['height']:.4f}"
        })
             
    if data_for_df:
        df = pd.DataFrame(data_for_df)
        
        if low_confidence_detected:
            st.warning(f"¡Atención! Se detectaron predicciones con confianza **inferior al {CONFIDENCE_THRESHOLD*100:.0f}%**. Estas podrían no ser precisas. Consulta la tabla de abajo para ver los valores exactos.")
        
        st.markdown("---")
        
        st.subheader("📊 Resumen Rápido de Detecciones (Todas las Confianzas)")
        high_confidence_df = df[df['Confianza (%)'] >= CONFIDENCE_THRESHOLD * 100]
        if not high_confidence_df.empty:
            cols = st.columns(len(high_confidence_df))
            for i, row in high_confidence_df.iterrows():
                cols[i % len(cols)].metric( 
                    label=row['Etiqueta'], 
                    value=f"{row['Confianza (%)']:.2f}%",
                    delta="Confianza"
                )
        else:
            st.info(f"No hay detecciones que superen el {CONFIDENCE_THRESHOLD*100:.0f}% de confianza para mostrar en este resumen.")


        st.markdown("---")
        
        st.subheader("📈 Nivel de Confianza por Marca Detectada")
        st.bar_chart(df, x='Etiqueta', y='Confianza (%)')

        st.markdown("---")
        
        st.subheader("🔍 Resultados Detallados y Bounding Box (Bbox)")
        
        def highlight_low_confidence(val):
            color = 'background-color: #f7a0a0' if val < CONFIDENCE_THRESHOLD * 100 else ''
            return color
        
        # --- CORRECCIÓN: Reemplazo de set_precision por .format() ---
        styled_df = df.style.applymap(
            highlight_low_confidence, 
            subset=['Confianza (%)']
        ).format({'Confianza (%)': '{:.2f}'}) 
        # ------------------------------------------------------------
        
        st.dataframe(styled_df, width=800)
        st.caption("Las filas sombreadas en rojo claro indican que la confianza es inferior al 50%.")
        
        high_confidence_labels = high_confidence_df['Etiqueta'].tolist()
        return high_confidence_labels if high_confidence_labels else ['No se detectaron marcas con alta confianza (>50%)']
    else:
        st.warning("El modelo no encontró ninguna detección. Por favor, asegúrate de que la imagen sea clara.")
        return None

if 'detection_complete' not in st.session_state:
    st.session_state.detection_complete = False

def reset_session():
    st.session_state.detection_complete = False
    if 'uploaded_file' in st.session_state:
        del st.session_state.uploaded_file
    st.rerun()

if __name__ == '__main__':
    
    st.set_page_config(page_title="Detector de Competidores McDonald's", layout="centered")

    st.markdown("""
    <style>
    .title-container {
        background-color: #DA291C;
        color: white; 
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 5px solid #FFC72C;
        margin-bottom: 30px;
    }
    .main-title, .subtitle, .stAlert p, .stAlert .stMarkdown {
        color: white !important; 
        text-align: center !important; 
    }
    
    h3 {
        color: #DA291C; 
        border-bottom: 2px solid #FFC72C;
        padding-bottom: 5px;
    }
    div.stButton > button {
        background-color: #FFC72C;
        color: #DA291C;
        font-weight: bold;
        font-size: 1.1em;
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid #DA291C;
    }
    div.stButton > button:hover {
        background-color: #DA291C;
        color: white;
        border: 2px solid #FFC72C;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="title-container">
        <p class="main-title">🍟 McDetector de Competidores 🍔</p>
        <p class="subtitle">Análisis de Publicidad de Comida Rápida (Selector de Iteración)</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Sube tu Imagen para el Análisis")

    selected_iteration = st.selectbox(
        "Elige la versión del modelo a usar:",
        AVAILABLE_ITERATIONS
    )
    iteration_name = selected_iteration.split(' ')[0] 

    if not st.session_state.detection_complete:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (JPG/PNG):",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), caption='Imagen Subida', width=400) 
            st.markdown("---")
            
            if st.button(f"🚀 Iniciar Detección con {iteration_name}"):
                st.session_state.detection_complete = True
                st.session_state.uploaded_file = uploaded_file
                st.session_state.iteration_to_use = iteration_name
                st.rerun() 
    
    else:
        uploaded_file = st.session_state.uploaded_file
        iteration_to_use = st.session_state.iteration_to_use
        
        FULL_ENDPOINT = f"{BASE_ENDPOINT_URL}{iteration_to_use}/image"

        st.subheader("🖼️ Imagen Analizada")
        st.image(Image.open(uploaded_file), caption=uploaded_file.name, width=400)
        
        with st.container():
            image_bytes = uploaded_file.getvalue()
            results_df = clasificar_imagen(image_bytes, uploaded_file.name, FULL_ENDPOINT, iteration_to_use) 
            
            if results_df:
                st.balloons() 
                st.markdown(f"**Marcas Detectadas (Alta Confianza):** **{', '.join([m for m in results_df if 'No se detectaron' not in m])}**")
                if 'No se detectaron' in results_df:
                    st.info("Revisa la tabla de detalles; solo se encontraron detecciones de baja confianza.")
        
        st.markdown("---")
        
        if st.button("⬅️ Realizar Otra Detección"):
            reset_session()
            
    st.markdown("---")
    st.caption(f"Los resultados de la predicción se registran en el archivo `{CSV_FILE}`. **Modelo Usado:** {iteration_name}")