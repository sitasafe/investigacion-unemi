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
        "dispersion": "🎯 Gráfico de Correlación",
        "encuesta": "📝 Simulación de Encuesta",
        "btn_reg": "Registrar y Actualizar",
        "exito": "¡Datos enviados con éxito!",
        "descarga": "📥 Descargar Base CSV",
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
# CONFIGURACIÓN Y ESTILO
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

# ------------------------------------------------
# MANEJO DE ESTADO DE DATOS
# ------------------------------------------------
with st.sidebar:
    st.header(lang["config"])
    n_muestra_input = st.slider(lang["muestra"], 50, 500, 100)

if 'main_data' not in st.session_state:
    np.random.seed(42)
    st.session_state.main_data = pd.DataFrame({
        "Uso_IA": np.random.choice(["Andamiaje","Sustituto"], n_muestra_input, p=[0.65,0.35]),
        "Analisis": np.random.normal(3.8,0.5,n_muestra_input).clip(1,5),
        "Evaluacion": np.random.normal(3.2,0.6,n_muestra_input).clip(1,5),
        "Autorregulacion": np.random.normal(3.5,0.4,n_muestra_input).clip(1,5),
        "Inferencia": np.random.normal(3.9,0.3,n_muestra_input).clip(1,5)
    })

datos = st.session_state.main_data
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]

# ------------------------------------------------
# TABS (UN SOLO ICONO POR PESTAÑA)
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab1:
    st.header(lang["tabs"][0])
    st.write(f"{'Relazione Accademica' if sel_idioma=='Italiano' else 'Informe Académico'} estructurado bajo normas **APA 7ma edición**.")
    st.info(lang["obj"])

with tab2:
    st.header(lang["tabs"][1])
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    c1, c2, c3 = st.columns(3)
    c1.metric(lang["m_est"], len(datos))
    c2.metric(lang["m_and"], f"{uso.get('Andamiaje',0):.1f}%")
    c3.metric(lang["m_sus"], f"{uso.get('Sustituto',0):.1f}%", delta_color="inverse")
    fig_pie = px.pie(names=uso.index, values=uso.values, hole=0.4, color_discrete_sequence=['#BEE3DB', '#FFD8BE'])
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.header(lang["tabs"][2])
    promedios = datos[habilidades].mean()
    df_bloom = pd.DataFrame({"Habilidad": habilidades, "Promedio": promedios.values})
    fig_bar = px.bar(df_bloom, x="Habilidad", y="Promedio", color="Promedio", color_continuous_scale='Teal')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.header(lang["tabs"][3])
    st.subheader("Intervalos de Confianza (95%)" if sel_idioma=="Español" else "Intervalli di confidenza (95%)")
    std = datos[habilidades].std()
    error = 1.96*(std/np.sqrt(len(datos)))
    df_ic = pd.DataFrame({
        "Habilidad": habilidades, 
        "Media": promedios.values, 
        "IC Inferior": promedios.values - error.values, 
        "IC Superior": promedios.values + error.values
    })
    st.dataframe(df_ic.style.format(precision=4).bar(subset=['Media'], color='#BEE3DB', vmin=1, vmax=5).highlight_max(subset=['Media'], color='#FFD8BE'), use_container_width=True)
    st.subheader(lang["regresion"])
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    X = sm.add_constant(datos["Uso_IA_bin"])
    modelo = sm.OLS(datos["Indice_PC"], X).fit()
    st.text(f"{'Coefficiente R²' if sel_idioma=='Italiano' else 'Coeficiente R²'}: {modelo.rsquared:.4f}")
    st.subheader(lang["dispersion"])
    labels_graf = {"Analisis": "Análisis" if sel_idioma=="Español" else "Analisi", "Indice_PC": "Pensamiento Crítico" if sel_idioma=="Español" else "Pensiero Critico", "Uso_IA": "Uso IA"}
    fig_disp = px.scatter(datos, x="Analisis", y="Indice_PC", color="Uso_IA", trendline="ols", labels=labels_graf, color_discrete_sequence=['#BEE3DB', '#FFD8BE'])
    st.plotly_chart(fig_disp, use_container_width=True)

with tab5:
    st.header(lang["tabs"][4])
    with st.expander(lang["guia_doc"], expanded=True):
        st.write(lang["txt_doc"])
    with st.expander(lang["guia_est"]):
        st.write(lang["txt_est"])
    with st.expander(lang["guia_ins"]):
        st.write(lang["txt_ins"])

st.divider()
st.header(lang["encuesta"])
with st.form("encuesta"):
    u_sel = st.selectbox(lang["modo_uso"], [lang["m_and"], lang["m_sus"]])
    if st.form_submit_button(lang["btn_reg"]):
        st.session_state.lanzar_globos = True
        val_uso = "Andamiaje" if u_sel in ["Andamiaje", "Impalcatura"] else "Sustituto"
        nuevo_dato = pd.DataFrame({
            "Uso_IA": [val_uso],
            "Analisis": [np.random.normal(4.0, 0.4)],
            "Evaluacion": [np.random.normal(3.5, 0.5)],
            "Autorregulacion": [np.random.normal(3.8, 0.3)],
            "Inferencia": [np.random.normal(4.1, 0.2)]
        })
        st.session_state.main_data = pd.concat([st.session_state.main_data, nuevo_dato], ignore_index=True)
        st.rerun()

st.download_button(lang["descarga"], datos.to_csv(index=False), "datos_unemi.csv", "text/csv")

# ------------------------------------------------
# PIE DE PÁGINA (CON CAPTION UNIFICADO)
# ------------------------------------------------
st.divider()
st.caption("""
Research Data Analytics System  
Developed by Ing. Willan E. Álvarez C.  
Maestría en Educación mención en Docencia e Investigación en Educación Superior – Universidad Estatal de Milagro
""")
