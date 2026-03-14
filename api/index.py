import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

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
st.title("📊 Informe Académico de Investigación")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico")

st.markdown("""
**🎓 Maestría en Educación UNEMI** | **👨‍🏫 Tutor:** Bonisoli Lorenzo PhD. | **📅 Fecha:** 07/03/2026
""")

st.write("### 👥 Equipo de Investigación")
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
# LÓGICA DE DATOS
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n_muestra = st.slider("Tamaño de la muestra", 50, 500, 100)

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

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Producto", "📊 Diagnóstico", "🧠 Mapeo", "📉 Estadística", "💡 Guía"])

with tab1:
    st.header("📄 Producto Esperado")
    st.write("Informe Académico estructurado bajo normas **APA 7ma edición**.")
    st.info("Objetivo: Analizar la relación entre IAGen y Pensamiento Crítico en UNEMI.")

with tab2:
    st.header("📈 Diagnóstico Situacional")
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("Muestra", n_muestra)
    c2.metric("Andamiaje", f"{uso.get('Andamiaje',0):.1f}%")
    c3.metric("Sustituto", f"{uso.get('Sustituto',0):.1f}%", delta_color="inverse")
    
    fig_pie = px.pie(names=uso.index, values=uso.values, hole=0.4, color_discrete_sequence=['#BEE3DB', '#FFD8BE'])
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.header("🧠 Mapeo de Influencia Cognitiva")
    promedios = datos[habilidades].mean()
    df_bloom = pd.DataFrame({"Habilidad": habilidades, "Promedio": promedios.values})
    fig_bar = px.bar(df_bloom, x="Habilidad", y="Promedio", color="Promedio", color_continuous_scale='Teal')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.header("🔍 Análisis Estadístico")
    st.subheader("Intervalos de Confianza (95%)")
    std = datos[habilidades].std()
    error = 1.96*(std/np.sqrt(n_muestra))
    df_ic = pd.DataFrame({"Habilidad":habilidades, "Media":promedios.values, "IC Inferior":promedios-error, "IC Superior":promedios+error})
    st.dataframe(df_ic, use_container_width=True)

    st.subheader("🤖 Modelo de Regresión")
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    X = sm.add_constant(datos["Uso_IA_bin"])
    modelo = sm.OLS(datos["Indice_PC"], X).fit()
    st.text(f"Coeficiente de determinación (R²): {modelo.rsquared:.4f}")
    st.toast("Estadísticos actualizados", icon="📈")

with tab5:
    st.header("💡 Guía de Recomendaciones")
    with st.expander("👨‍🏫 Para Docentes", expanded=True):
        st.write("- Fomentar el uso de IA como **andamiaje cognitivo**.")
        st.write("- Evaluar la capacidad de validación crítica del alumno.")
    with st.expander("🎓 Para Estudiantes"):
        st.write("- Contrastar resultados de IA con fuentes académicas.")
        st.write("- No sustituir el juicio propio por la respuesta del bot.")
    with st.expander("🏛️ Para Instituciones"):
        st.write("- Crear políticas de integridad académica frente a la IAGen.")

st.divider()
st.header("📝 Simulación de Encuesta")
with st.form("encuesta"):
    u_sel = st.selectbox("Modo de uso", ["Andamiaje", "Sustituto"])
    if st.form_submit_button("Registrar"):
        st.balloons()
        st.success("¡Datos enviados con éxito!")

st.download_button("📥 Descargar Base CSV", datos.to_csv(index=False), "datos_unemi.csv", "text/csv")
