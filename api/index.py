import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ------------------------------------------------
# TRADUCCIONES (DICCIONARIO)
# ------------------------------------------------
traducciones = {
    "Español": {
        "titulo": "📊 Informe Académico de Investigación",
        "sub": "Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico",
        "tutor": "Tutor",
        "equipo": "👥 Equipo de Investigación",
        "config": "⚙️ Configuración",
        "muestra": "Tamaño de la muestra",
        "tabs": ["📂 Producto", "📊 Diagnóstico", "🧠 Mapeo", "📉 Estadística", "💡 Guía"],
        "m_estudiantes": "Estudiantes",
        "m_andamiaje": "Andamiaje",
        "m_sustituto": "Sustituto",
        "r2": "Coeficiente de determinación",
        "btn_descarga": "📥 Descargar Base CSV",
        "guia_doc": "👨‍🏫 Para Docentes",
        "guia_est": "🎓 Para Estudiantes",
        "guia_inst": "🏛️ Para Instituciones"
    },
    "Italiano": {
        "titulo": "📊 Rapporto Accademico di Ricerca",
        "sub": "Impatto dell'Intelligenza Artificiale Generativa sul Pensiero Critico",
        "tutor": "Tutore",
        "equipo": "👥 Team di Ricerca",
        "config": "⚙️ Impostazioni",
        "muestra": "Dimensione del campione",
        "tabs": ["📂 Prodotto", "📊 Diagnosi", "🧠 Mappatura", "📉 Statistica", "💡 Guida"],
        "m_estudiantes": "Studenti",
        "m_andamiaje": "Impalcatura",
        "m_sustituto": "Sostituto",
        "r2": "Coefficiente di determinazione",
        "btn_descarga": "📥 Scarica Database CSV",
        "guia_doc": "👨‍🏫 Per i Docenti",
        "guia_est": "🎓 Per gli Studenti",
        "guia_inst": "🏛️ Per le Istituzioni"
    }
}

# ------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

# Selector de idioma en la barra lateral
with st.sidebar:
    st.header("🌐 Lingua / Idioma")
    idioma = st.radio("Seleccione idioma:", ["Español", "Italiano"], horizontal=True)
    lang = traducciones[idioma]

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

st.markdown(f"**🎓 Maestría en Educación UNEMI** | **👨‍🏫 {lang['tutor']}:** Bonisoli Lorenzo PhD. | **📅 07/03/2026**")

st.write(f"### {lang['equipo']}")
c_i1, c_i2, c_i3 = st.columns(3)
integrantes = [
    "👨‍💻 Willan Efrén Álvarez Carmona", "👩‍🏫 Tania Jacqueline Barcos Villalva",
    "👩‍🔬 Selene Anaís Guagua Valencia", "👨‍💼 Pedro Javier Figueroa Vergara",
    "👩‍🎓 Nohemí Nicole Miranda Jiménez"
]
with c_i1:
    st.markdown(f'<div class="integrante-card">{integrantes[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="integrante-card">{integrantes[1]}</div>', unsafe_allow_html=True)
with c_i2:
    st.markdown(f'<div class="integrante-card">{integrantes[2]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="integrante-card">{integrantes[3]}</div>', unsafe_allow_html=True)
with c_i3:
    st.markdown(f'<div class="integrante-card">{integrantes[4]}</div>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------
# LÓGICA DE DATOS
# ------------------------------------------------
with st.sidebar:
    st.header(lang["config"])
    n_muestra = st.slider(lang["muestra"], 50, 500, 100)

@st.cache_data
def generar_datos(n):
    np.random.seed(42)
    df = pd.DataFrame({
        "Uso_IA": np.random.choice(["Andamiaje","Sustituto"], n, p=[0.65,0.35]),
        "Analisis": np.random.normal(3.8,0.5,n).clip(1,5),
        "Evaluacion": np.random.normal(3.2,0.6,n).clip(1,5),
        "Autorregulacion": np.random.normal(3.5,0.4,n).clip(1,5),
        "Inferencia": np.random.normal(3.9,0.3,n).clip(1,5)
    })
    return df

datos = generar_datos(n_muestra)
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
promedios = datos[habilidades].mean()

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab1:
    st.header(lang["tabs"][0])
    txt_prod = "Rapporto accademico strutturato secondo le norme **APA 7ma edizione**." if idioma == "Italiano" else "Informe Académico estructurado bajo normas **APA 7ma edición**."
    st.write(txt_prod)

with tab2:
    st.header(lang["tabs"][1])
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    c1, c2, c3 = st.columns(3)
    c1.metric(lang["m_estudiantes"], n_muestra)
    c2.metric(lang["m_andamiaje"], f"{uso.get('Andamiaje',0):.1f}%")
    c3.metric(lang["m_sustituto"], f"{uso.get('Sustituto',0):.1f}%", delta_color="inverse")
    
    fig_pie = px.pie(names=uso.index, values=uso.values, hole=0.4, color_discrete_sequence=['#BEE3DB', '#FFD8BE'])
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.header(lang["tabs"][2])
    df_bloom = pd.DataFrame({"Habilidad": habilidades, "Promedio": promedios.values})
    fig_bar = px.bar(df_bloom, x="Habilidad", y="Promedio", color="Promedio", color_continuous_scale='Teal')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.header(lang["tabs"][3])
    std = datos[habilidades].std()
    error = 1.96*(std/np.sqrt(n_muestra))
    df_ic = pd.DataFrame({
        "Habilidad": habilidades, 
        "Media": promedios.values, 
        "IC Inferior": promedios.values - error.values, 
        "IC Superior": promedios.values + error.values
    })
    st.dataframe(
        df_ic.style.format(precision=4)
        .bar(subset=['Media'], color='#BEE3DB', vmin=1, vmax=5)
        .highlight_max(subset=['Media'], color='#FFD8BE'), 
        use_container_width=True
    )

with tab5:
    st.header(lang["tabs"][4])
    with st.expander(lang["guia_doc"], expanded=True):
        st.write("• Promuovere l'uso dell'IA come impalcatura cognitiva." if idioma == "Italiano" else "• Fomentar el uso de IA como andamiaje cognitivo.")
    with st.expander(lang["guia_est"]):
        st.write("• Valutare criticamente i risultati dell'IA." if idioma == "Italiano" else "• Evaluar críticamente los resultados de la IA.")
    with st.expander(lang["guia_inst"]):
        st.write("• Creare politiche di integrità accademica." if idioma == "Italiano" else "• Crear políticas de integridad académica.")

st.divider()
st.download_button(lang["btn_descarga"], datos.to_csv(index=False), "datos_unemi.csv", "text/csv")
