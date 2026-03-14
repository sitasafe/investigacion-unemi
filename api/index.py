import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import google.generativeai as genai

# ------------------------------------------------
# CONFIGURACIÓN DE GEMINI (IA)
# ------------------------------------------------
API_KEY = "AIzaSyC3XWlImuVEFuqo5p0H0FjcKX0n5XmlF1E"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ------------------------------------------------
# 1. FUNCIONES DE LÓGICA
# ------------------------------------------------
@st.cache_data
def generar_datos(n):
    np.random.seed(42)
    return pd.DataFrame({
        "Uso_IA": np.random.choice(["Andamiaje","Sustituto"], n, p=[0.65,0.35]),
        "Analisis": np.random.normal(3.8,0.5,n).clip(1,5),
        "Evaluacion": np.random.normal(3.2,0.6,n).clip(1,5),
        "Autorregulacion": np.random.normal(3.5,0.4,n).clip(1,5),
        "Inferencia": np.random.normal(3.9,0.3,n).clip(1,5)
    })

def obtener_explicacion_ia(df_stats, r2):
    try:
        prompt = f"""
        Actúa como un experto en estadística educativa de la UNEMI. 
        Analiza estos datos sobre IA y Pensamiento Crítico:
        - Promedios por habilidad: {df_stats.to_dict()}
        - Coeficiente R2: {r2:.4f}
        
        Dame una conclusión breve (máximo 3 párrafos) sobre si la IA está ayudando 
        al pensamiento crítico o si actúa como un sustituto. Sé muy profesional.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Nota: La IA está procesando otros datos. ({str(e)})"

# ------------------------------------------------
# 2. DICCIONARIO DE TRADUCCIÓN
# ------------------------------------------------
idiomas = {
    "Español": {
        "titulo": "📊 Informe Académico de Investigación",
        "sub": "Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico",
        "tutor": "Tutor",
        "equipo": "👥 Equipo de Investigación",
        "config": "⚙️ Configuración",
        "muestra": "Tamaño de la muestra",
        "tabs": ["📂 Producto", "📊 Diagnóstico", "🧠 Mapeo", "📉 Estadística", "💡 Guía"],
        "obj": "Objetivo: Analizar la relación entre Inteligencia Artificial Generativa y Pensamiento Crítico en la UNEMI.",
        "m_est": "Estudiantes",
        "m_and": "Andamiaje",
        "m_sus": "Sustituto",
        "regresion": "🤖 Modelo de Regresión",
        "dispersion": "🎯 Gráfico de Correlación",
        "encuesta": "📝 Simulación de Encuesta",
        "btn_reg": "Registrar y Actualizar",
        "exito": "¡Datos enviados con éxito!",
        "descarga": "📥 Descargar",
        "modo_uso": "Modo de uso",
        "guia_doc": "👨‍🏫 Para Docentes",
        "guia_est": "🎓 Para Estudiantes",
        "guia_ins": "🏛️ Para Instituciones",
        "txt_doc": "- Fomentar el uso de IA como **andamiaje cognitivo**.",
        "txt_est": "- Contrastar resultados de IA con fuentes académicas.",
        "txt_ins": "- Crear políticas de integridad académica y ética digital."
    },
    "Italiano": {
        "titulo": "📊 Rapporto Accademico di Ricerca",
        "sub": "Impatto dell'Intelligenza Artificiale Generativa sul Pensiero Critico",
        "tutor": "Tutore",
        "equipo": "👥 Team di Ricerca",
        "config": "⚙️
