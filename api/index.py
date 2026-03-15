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
        return "⚠️ Error: GEMINI_API_KEY no detectada en los Secrets."
    try:
        prompt = f"""
        Actúa como un experto en estadística educativa de la UNEMI. 
        Analiza: Promedios {df_stats.to_dict()}, R²: {r2:.4f}, Pearson: {pearson:.3f}, p-value: {p_val:.4f}.
        Concluye en 3 párrafos sobre el impacto de la IA en el pensamiento crítico.
        """
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "Análisis no disponible."
    except Exception:
        # PLAN B de seguridad si falla la cuota de la IA
        interpretacion = "Andamiaje Cognitivo" if pearson > 0.3 else "Sustitución de procesos"
        return f"**Análisis Automático (Plan B):** Tendencia hacia el **{interpretacion}**. Relación significativa: {'Sí' if p_val < 0.05 else 'No'} (p={p_val:.4f})."

# ------------------------------------------------
# 2. DICCIONARIO DE TRADUCCIÓN COMPLETO
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
        "descarga": "📥 Descargar database CSV",
        "modo_uso": "Modo de uso",
        "guia_doc": "👨‍🏫 Para Docentes",
        "guia_est": "🎓 Para Estudiantes",
        "guia_ins": "🏛️ Para Instituciones",
        "txt_doc": "- Fomentar el uso de IA como **andamiaje cognitivo**.",
        "txt_est": "- Contrastar resultados de IA con fuentes académicas.",
        "txt_ins": "- Crear políticas de integridad académica y ética digital."
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
        "descarga": "📥 Scarica il database CSV",
        "modo_uso": "Modalità d'uso",
        "guia_doc": "👨‍🏫 Per i docenti",
        "guia_est": "🎓 Per gli studenti",
        "guia_ins": "🏛️ Per le istituzioni",
        "txt_doc": "- Incoraggiare l'uso dell'IA come **impalcatura cognitiva**.",
        "txt_est": "- Confrontare i risultati dell'IA con fonti accademiche.",
        "txt_ins": "- Creare politiche di integrità accademica e etica digitale."
    }
}

# ------------------------------------------------
# 3. CONFIGURACIÓN Y ESTILO CSS
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

# Lógica de globos
if st.session_state.get('lanzar_globos'):
    st.balloons()
    st.session_state.lanzar_globos = False

st.markdown("""
<style>
    .stApp { background-color: #FDFCF0; }
    .integrante-card {
        background-color: #FFFFFF; padding: 10px; border-radius: 8px; 
        border-left: 5px solid #BEE3DB; margin-bottom: 8px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .ia-box {
        background-color: #E8F0FE; padding: 20px; border-radius: 10px;
        border-left: 5px solid #4285F4; margin: 10px 0; border: 1px solid #D1D9E6;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 4. SIDEBAR TRADUCIDO
# ------------------------------------------------
with st.sidebar:
    st.header("🌐 Language / Lingua")
    sel_idioma = st.radio("Selec.", ["Español", "Italiano"], label_visibility="collapsed")
    lang = idiomas[sel_idioma]
    
    st.divider()
    st.header(lang["config"])
    n_muestra_input = st.slider(lang["muestra"], 50, 500, 100)

# ------------------------------------------------
# 5. MANEJO DE DATOS
# ------------------------------------------------
if 'main_data' not in st.session_state or len(st.session_state.main_data) != n_muestra_input:
    st.session_state.main_data = generar_datos(n_muestra_input)

datos = st.session_state.main_data
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
promedios = datos[habilidades].mean()

# ------------------------------------------------
# 6. INTERFAZ PRINCIPAL
# ------------------------------------------------
st.title(lang["titulo"])
st.subheader(lang["sub"])
st.markdown(f"**👨‍🏫 {lang['tutor']}:** Bonisoli Lorenzo PhD. | **📅 07/03/2026**")

st.write(f"### {lang['equipo']}")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="integrante-card">👨‍💻 Willan Efrén Álvarez Carmona</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👩‍🏫 Tania Jacqueline Barcos Villalva</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="integrante-card">👩‍🔬 Selene Anaís Guagua Valencia</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👨‍💼 Pedro Javier Figueroa Vergara</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="integrante-card">👩‍🎓 Nohemí Nicole Miranda Jiménez</div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab1:
    st.header(lang["tabs"][0])
    st.info(lang["obj"])

with tab2:
    st.header(lang["tabs"][1])
    st.dataframe(datos[habilidades].describe().T)
    st.divider()
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    c1, c2, c3 = st.columns(3)
    c1.metric(lang["m_est"], len(datos))
    c2.metric(lang["m_and"], f"{uso.get('Andamiaje',0):.1f}%")
    c3.metric(lang["m_sus"], f"{uso.get('Sustituto',0):.1f}%")
    st.plotly_chart(px.pie(names=uso.index, values=uso.values, hole=0.4, color_discrete_sequence=['#BEE3DB', '#FFD8BE']), use_container_width=True)

with tab3:
    st.header(lang["tabs"][2])
    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.subheader("🧠 Radar Profile")
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=promedios.values, theta=habilidades, fill='toself', name='Promedio', fillcolor='rgba(190, 227, 219, 0.6)', line=dict(color='#BEE3DB')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1,5])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_bar:
        st.subheader("📊 Bar Comparison")
        st.plotly_chart(px.bar(pd.DataFrame({"Hab": habilidades, "Prom": promedios.values}), x="Hab", y="Prom", color="Prom", color_continuous_scale='Teal'), use_container_width=True)

with tab4:
    st.header(lang["tabs"][3])
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    modelo = sm.OLS(datos["Indice_PC"], sm.add_constant(datos["Uso_IA_bin"])).fit()
    correlacion, p_valor = stats.pearsonr(datos["Uso_IA_bin"], datos["Indice_PC"])
    
    st.subheader("🤖 AI Interpretation")
    if st.button("Generate Analysis / Genera Analisi"):
        with st.spinner("Analyzing..."):
            explicacion = obtener_explicacion_ia(promedios, modelo.rsquared, correlacion, p_valor)
            st.markdown(f'<div class="ia-box">{explicacion}</div>', unsafe_allow_html=True)
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{modelo.rsquared:.4f}")
    m2.metric("Pearson", f"{correlacion:.3f}")
    m3.metric("p-value", f"{p_valor:.4f}")

with tab5:
    st.header(lang["tabs"][4])
    for g in ["guia_doc", "guia_est", "guia_ins"]:
        with st.expander(lang[g]): st.write(lang["txt_" + g.split('_')[1]])

st.divider()
st.header("📝 Simulation / Simulazione")
with st.form("encuesta_form"):
    u_sel = st.selectbox(lang["modo_uso"], [lang["m_and"], lang["m_sus"]])
    if st.form_submit_button(lang["btn_reg"]):
        st.session_state.lanzar_globos = True
        st.rerun()

st.download_button(lang["descarga"], datos.to_csv(index=False), "investigacion_unemi.csv", "text/csv")
st.caption("Developed by Ing. Willan E. Álvarez C. - UNEMI 2026")
