import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import google.generativeai as genai  # <--- NUEVA LIBRERÍA

# ------------------------------------------------
# CONFIGURACIÓN DE GEMINI (IA)
# ------------------------------------------------
# Nota: En producción, usa st.secrets para mayor seguridad
API_KEY = "TU_API_KEY_AQUI" 
if API_KEY != "TU_API_KEY_AQUI":
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
    """Usa Gemini para interpretar los datos estadísticos."""
    try:
        prompt = f"""
        Actúa como un experto en estadística educativa. 
        Analiza estos datos de una investigación en la UNEMI sobre IA y Pensamiento Crítico:
        - Promedios por habilidad: {df_stats.to_string()}
        - Coeficiente R2 del modelo: {r2:.4f}
        
        Dame una conclusión breve (máximo 3 párrafos) sobre si la IA está ayudando 
        al pensamiento crítico o si actúa como un sustituto. Sé muy profesional.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Conecta tu API Key de Gemini para ver la interpretación automática de la IA."

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
        "config": "⚙️ Impostazioni",
        "muestra": "Dimensione del campione",
        "tabs": ["📂 Prodotto", "📊 Diagnosi", "🧠 Mappatura", "📉 Statistica", "💡 Guida"],
        "obj": "Obiettivo: Analizzare la relazione tra IAGen e Pensiero Critico in UNEMI.",
        "m_est": "Studenti",
        "m_and": "Impalcatura",
        "m_sus": "Sostituto",
        "regresion": "🤖 Modello di Regressione",
        "dispersion": "🎯 Grafico di Correlazione",
        "encuesta": "📝 Simulazione di Sondaggio",
        "btn_reg": "Registrare e Aggiornare",
        "exito": "Dati inviati con successo!",
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
# 3. CONFIGURACIÓN Y ESTILO
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

if st.session_state.get('lanzar_globos'):
    st.balloons()
    st.session_state.lanzar_globos = False

with st.sidebar:
    st.header("🌐 Lingua")
    sel_idioma = st.radio("Seleccione Idioma / Scegli la lingua", ["Español", "Italiano"], horizontal=True)
    lang = idiomas[sel_idioma]

st.markdown("""
<style>
    .stApp { background-color: #FDFCF0; }
    .integrante-card {
        background-color: #FFFFFF;
        padding: 10px; border-radius: 8px; border-left: 5px solid #BEE3DB;
        margin-bottom: 8px; box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
    }
    .ia-box {
        background-color: #E8F0FE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4285F4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 4. MANEJO DE DATOS
# ------------------------------------------------
with st.sidebar:
    st.header(lang["config"])
    n_muestra_input = st.slider(lang["muestra"], 50, 500, 100)

if 'main_data' not in st.session_state or len(st.session_state.main_data) != n_muestra_input:
    st.session_state.main_data = generar_datos(n_muestra_input)

datos = st.session_state.main_data
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
promedios = datos[habilidades].mean()

# ------------------------------------------------
# 5. INTERFAZ
# ------------------------------------------------
st.title(lang["titulo"])
st.subheader(lang["sub"])
st.markdown(f"**🎓 Maestría en Educación mención en Docencia e Investigación en Educación Superior** | **👨‍🏫 {lang['tutor']}:** Bonisoli Lorenzo PhD. | **📅 Fecha:** 07/03/2026")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab1:
    st.header(lang["tabs"][0])
    st.write(f"{'Relazione Accademica' if sel_idioma=='Italiano' else 'Informe Académico'} (Normas APA 7ma).")
    st.info(lang["obj"])
    st.subheader("📝 Metodología")
    st.markdown("""
    **Tipo de estudio:** Cuantitativo exploratorio | **Diseño:** No experimental transversal  
    **Instrumento:** Escala Likert 1-5 | **Variables:** Uso de IA y Pensamiento Crítico.
    """)

with tab2:
    st.header(lang["tabs"][1])
    st.subheader("🔎 Validación de Datos")
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
    col_rad1, col_rad2 = st.columns(2)
    with col_rad1:
        st.plotly_chart(px.bar(pd.DataFrame({"Hab": habilidades, "Prom": promedios.values}), x="Hab", y="Prom", color="Prom", color_continuous_scale='Teal'), use_container_width=True)
    with col_rad2:
        st.subheader("🧠 Perfil Radar")
        fig_radar = go.Figure(data=go.Scatterpolar(r=promedios.values, theta=habilidades, fill='toself', fillcolor='rgba(190, 227, 219, 0.6)', line=dict(color='#BEE3DB')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1,5])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

with tab4:
    st.header(lang["tabs"][3])
    st.subheader("🔢 Correlación de Pearson")
    st.plotly_chart(px.imshow(datos[habilidades].corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
    st.divider()
    
    # REGRESIÓN
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    modelo = sm.OLS(datos["Indice_PC"], sm.add_constant(datos["Uso_IA_bin"])).fit()
    
    st.subheader("🤖 Interpretación con IA (Google Gemini)")
    if st.button("Generar Análisis con IA"):
        with st.spinner("Gemini está analizando los datos..."):
            explicacion = obtener_explicacion_ia(promedios, modelo.rsquared)
            st.markdown(f'<div class="ia-box">{explicacion}</div>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📘 Estadísticas de Regresión")
    if modelo.rsquared > 0.5: st.success(f"Relación fuerte (R²: {modelo.rsquared:.4f})")
    else: st.info(f"Relación moderada/débil (R²: {modelo.rsquared:.4f})")

with tab5:
    st.header(lang["tabs"][4])
    for g in ["guia_doc", "guia_est", "guia_ins"]:
        with st.expander(lang[g]): st.write(lang["txt_" + g.split('_')[1]])

st.divider()
st.header(lang["encuesta"])
with st.form("encuesta"):
    u_sel = st.selectbox(lang["modo_uso"], [lang["m_and"], lang["m_sus"]])
    if st.form_submit_button(lang["btn_reg"]):
        st.session_state.lanzar_globos = True
        val_uso = "Andamiaje" if u_sel in ["Andamiaje", "Impalcatura"] else "Sustituto"
        nuevo = pd.DataFrame({"Uso_IA": [val_uso], "Analisis": [np.random.normal(4.0, 0.4)], "Evaluacion": [np.random.normal(3.5, 0.5)], "Autorregulacion": [np.random.normal(3.8, 0.3)], "Inferencia": [np.random.normal(4.1, 0.2)]})
        st.session_state.main_data = pd.concat([st.session_state.main_data, nuevo], ignore_index=True)
        st.rerun()

st.download_button(lang["descarga"], datos.to_csv(index=False), "datos_profesional.csv", "text/csv")

st.divider()
st.caption("""
Research Data Analytics System  
Developed by Ing. Willan E. Álvarez C.  
Maestría en Educación mención en Docencia e Investigación en Educación Superior – Universidad Estatal de Milagro
""")
