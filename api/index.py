import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ------------------------------------------------
# DICCIONARIO DE TRADUCCIÓN COMPLETO
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
        "dispersion": "🎯 Correlación: Uso de IA vs. Pensamiento Crítico",
        "encuesta": "📝 Simulación de Encuesta",
        "btn_reg": "Registrar",
        "exito": "¡Datos enviados con éxito!",
        "descarga": "📥 Descargar Base CSV"
    },
    "Italiano": {
        "titulo": "📊 Rapporto Accademico di Ricerca",
        "sub": "Impatto dell'Intelligenza Artificiale Generativa sul Pensiero Critico",
        "tutor": "Tutore",
        "equipo": "👥 Team di Ricerca",
        "config": "⚙️ Impostazioni",
        "muestra": "Dimensione del campione",
        "tabs": ["📂 Prodotto", "📊 Diagnosi", "🧠 Mappatura", "📉 Statistica", "💡 Guida"],
        "obj": "Obiettivo: Analizzare la relazione tra IAGen e Pensiero Critico in UNEMI.",
        "m_est": "Studenti",
        "m_and": "Impalcatura",
        "m_sus": "Sostituto",
        "regresion": "🤖 Modello di Regressione",
        "dispersion": "🎯 Correlazione: Uso IA vs. Pensiero Critico",
        "encuesta": "📝 Simulazione di Sondaggio",
        "btn_reg": "Registrare",
        "exito": "Dati inviati con successo!",
        "descarga": "📥 Scarica il database CSV"
    }
}

# ------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

with st.sidebar:
    st.header("🌐 Lingua")
    sel_idioma = st.radio("Seleccione Idioma / Scegli la lingua", ["Español", "Italiano"], horizontal=True)
    lang = idiomas[sel_idioma]

st.markdown("""
<style>
    .stApp { background-color: #FDFCF0; }
    .integrante-card {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #BEE3DB;
        margin-bottom: 8px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# PORTADA
# ------------------------------------------------
st.title(lang["titulo"])
st.subheader(lang["sub"])
st.markdown(f"**🎓 Maestría en Educación UNEMI** | **👨‍🏫 {lang['tutor']}:** Bonisoli Lorenzo PhD. | **📅 Fecha:** 07/03/2026")

st.write(f"### {lang['equipo']}")
c_i1, c_i2, c_i3 = st.columns(3)
with c_i1:
    st.markdown('<div class="integrante-card">👨‍💻 Willan Efrén Álvarez Carmona</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👩‍🏫 Tania Jacqueline Barcos Villalva</div>', unsafe_allow_html=True)
with c_i2:
    st.markdown('<div class="integrante-card">👩‍🔬 Selene Anaís Guagua Valencia</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👨‍💼 Pedro Javier Figueroa Vergara</div>', unsafe_allow_html=True)
with c_i3:
    st.markdown('<div class="integrante-card">👩‍🎓 Nohemí Nicole Miranda Jiménez</div>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------
# LÓG
