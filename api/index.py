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
import os

# ------------------------------------------------
# CONFIGURACIÓN DE GEMINI (IA)
# ------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

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

def obtener_explicacion_ia(df_stats, r2, pearson, p_val):
    if model is None:
        return "⚠️ Error: GEMINI_API_KEY no detectada."
    try:
        prompt = f"Analiza estos datos estadísticos: {df_stats.to_dict()}, R2:{r2:.4f}, Pearson:{pearson:.3f}, p:{p_val:.4f}. Concluye sobre IA como andamiaje o sustituto."
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "Análisis no disponible."
    except:
        return f"Análisis Automático: Tendencia a {'Andamiaje' if pearson > 0.3 else 'Sustitución'}. Significancia: {'Sí' if p_val < 0.05 else 'No'} (p={p_val:.4f})."

# ------------------------------------------------
# 2. DICCIONARIO DE TRADUCCIÓN (CORREGIDO Y AMPLIADO)
# ------------------------------------------------
idiomas = {
    "Español": {
        "sidebar_idioma": "🌐 Idioma",
        "seleccione": "Seleccione Idioma",
        "titulo": "📊 Informe Académico de Investigación",
        "sub": "Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico",
        "tutor": "Tutor",
        "equipo": "👥 Equipo de Investigación",
        "config": "⚙️ Configuración",
        "muestra": "Tamaño de la muestra",
        "tabs": ["📂 Producto", "📊 Diagnóstico", "🧠 Mapeo", "📉 Estadística", "💡 Guía"],
        "obj": "Objetivo: Analizar la relación entre IA Generativa y Pensamiento Crítico en la UNEMI.",
        "m_est": "Estudiantes",
        "m_and": "Andamiaje",
        "m_sus": "Sustituto",
        "btn_reg": "Registrar y Actualizar",
        "descarga": "📥 Descargar CSV",
        "modo_uso": "Modo de uso",
        "guia_doc": "👨‍🏫 Para Docentes",
        "guia_est": "🎓 Para Estudiantes",
        "guia_ins": "🏛️ Para Instituciones",
        "txt_doc": "- Fomentar el uso de IA como andamiaje cognitivo.",
        "txt_est": "- Contrastar resultados de IA con fuentes académicas.",
        "txt_ins": "- Crear políticas de integridad académica."
    },
    "Italiano": {
        "sidebar_idioma": "🌐 Lingua",
        "seleccione": "Seleziona Lingua",
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
        "btn_reg": "Registrare e Aggiornare",
        "descarga": "📥 Scarica CSV",
        "modo_uso": "Modalità d'uso",
        "guia_doc": "👨‍🏫 Per i docenti",
        "guia_est": "🎓 Per gli studenti",
        "guia_ins": "🏛️ Per le istituzioni",
        "txt_doc": "- Incoraggiare l'uso dell'IA come impalcatura cognitiva.",
        "txt_est": "- Confrontare i risultati dell'IA con fonti accademiche.",
        "txt_ins": "- Creare politiche di integrità accademica."
    }
}

# ------------------------------------------------
# 3. CONFIGURACIÓN Y SIDEBAR (TRADUCCIÓN DINÁMICA)
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

# Inicializar idioma en session_state para evitar saltos
if 'idioma_idx' not in st.session_state:
    st.session_state.idioma_idx = 0

with st.sidebar:
    # Usamos un radio temporal para capturar el cambio
    sel_temp = st.radio("Language / Idioma", ["Español", "Italiano"], index=st.session_state.idioma_idx)
    st.session_state.idioma_idx = 0 if sel_temp == "Español" else 1
    
    lang = idiomas[sel_temp]
    
    st.divider()
    st.header(lang["config"])
    n_muestra_input = st.slider(lang["muestra"], 50, 500, 100)

# ------------------------------------------------
# 4. MANEJO DE DATOS E INTERFAZ
# ------------------------------------------------
if 'main_data' not in st.session_state or len(st.session_state.main_data) != n_muestra_input:
    st.session_state.main_data = generar_datos(n_muestra_input)

datos = st.session_state.main_data
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
promedios = datos[habilidades].mean()

# --- TÍTULOS ---
st.title(lang["titulo"])
st.subheader(lang["sub"])
st.markdown(f"**👨‍🏫 {lang['tutor']}:** Bonisoli Lorenzo PhD. | **📅 07/03/2026**")

# --- EQUIPO ---
st.write(f"### {lang['equipo']}")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div style="background:#fff;padding:10px;border-radius:8px;border-left:5px solid #BEE3DB;margin-bottom:8px;">👨‍💻 Willan Efrén Álvarez Carmona</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#fff;padding:10px;border-radius:8px;border-left:5px solid #BEE3DB;">👩‍🏫 Tania Jacqueline Barcos Villalva</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div style="background:#fff;padding:10px;border-radius:8px;border-left:5px solid #BEE3DB;margin-bottom:8px;">👩‍🔬 Selene Anaís Guagua Valencia</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#fff;padding:10px;border-radius:8px;border-left:5px solid #BEE3DB;">👨‍💼 Pedro Javier Figueroa Vergara</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div style="background:#fff;padding:10px;border-radius:8px;border-left:5px solid #BEE3DB;">👩‍🎓 Nohemí Nicole Miranda Jiménez</div>', unsafe_allow_html=True)

st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab1:
    st.info(lang["obj"])

with tab2:
    st.dataframe(datos[habilidades].describe().T)
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(lang["m_est"], len(datos))
    col_m2.metric(lang["m_and"], f"{uso.get('Andamiaje',0):.1f}%")
    col_m3.metric(lang["m_sus"], f"{uso.get('Sustituto',0):.1f}%")
    st.plotly_chart(px.pie(names=uso.index, values=uso.values, hole=0.4), use_container_width=True)

with tab3:
    col_radar, col_bar = st.columns(2)
    with col_radar:
        fig_radar = go.Figure(data=go.Scatterpolar(r=promedios.values, theta=habilidades, fill='toself', name='Promedio'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1,5])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_bar:
        st.plotly_chart(px.bar(pd.DataFrame({"Hab": habilidades, "Prom": promedios.values}), x="Hab", y="Prom", color="Prom"), use_container_width=True)

with tab4:
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    modelo = sm.OLS(datos["Indice_PC"], sm.add_constant(datos["Uso_IA_bin"])).fit()
    corr, pval = stats.pearsonr(datos["Uso_IA_bin"], datos["Indice_PC"])
    
    if st.button("AI Analysis / Analisi AI"):
        with st.spinner("..."):
            st.write(obtener_explicacion_ia(promedios, modelo.rsquared, corr, pval))
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{modelo.rsquared:.4f}")
    m2.metric("Pearson", f"{corr:.3f}")
    m3.metric("p-value", f"{pval:.4f}")

with tab5:
    for g in ["guia_doc", "guia_est", "guia_ins"]:
        with st.expander(lang[g]): st.write(lang["txt_" + g.split('_')[1]])

st.caption("Developed by Ing. Willan E. Álvarez C. - UNEMI 2026")
