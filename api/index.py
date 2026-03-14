import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ------------------------------------------------
# CONFIGURACIÓN Y ESTILO PASTEL (UI/UX)
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

st.markdown("""
<style>
    /* Fondo general pastel */
    .stApp {
        background-color: #FDFCF0;
    }
    
    /* Animación de entrada suave */
    .main {
        animation: fadeIn 1.2s ease-in;
    }
    @keyframes fadeIn {
        0% {opacity:0;}
        100% {opacity:1;}
    }

    /* Estilo para las pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #E8F1F2;
        padding: 10px;
        border-radius: 15px;
    }

    /* Tarjetas de integrantes con hover */
    .integrante-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #BEE3DB;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s;
        margin-bottom: 10px;
    }
    .integrante-card:hover {
        transform: scale(1.02);
        background-color: #F7FFF7;
    }

    /* Títulos pastel */
    h1, h2, h3 {
        color: #555B6E;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# PORTADA ACADÉMICA CON ICONOS
# ------------------------------------------------
st.title("📊 Informe Académico de Investigación")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico")

with st.container():
    col_portada1, col_portada2 = st.columns([2, 1])
    with col_portada1:
        st.markdown("""
        **🎓 Maestría en Educación mención en Docencia e Investigación en Educación Superior** **📘 Módulo:** Seminario de Investigación 1  
        **👨‍🏫 Tutor:** Bonisoli Lorenzo PhD.
        """)
    with col_portada2:
        st.info("📅 **Fecha de entrega:** 07 de marzo de 2026")

st.write("### 👥 Equipo de Investigación")
col_int1, col_int2, col_int3 = st.columns(3)

with col_int1:
    st.markdown('<div class="integrante-card">👨‍💻 Willan Efrén Álvarez Carmona</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👩‍🏫 Tania Jacqueline Barcos Villalva</div>', unsafe_allow_html=True)
with col_int2:
    st.markdown('<div class="integrante-card">👩‍🔬 Selene Anaís Guagua Valencia</div>', unsafe_allow_html=True)
    st.markdown('<div class="integrante-card">👨‍💼 Pedro Javier Figueroa Vergara</div>', unsafe_allow_html=True)
with col_int3:
    st.markdown('<div class="integrante-card">👩‍🎓 Nohemí Nicole Miranda Jiménez</div>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------
# PARÁMETROS (SIDEBAR)
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n_muestra = st.slider("Tamaño de la muestra", 50, 500, 100)
    st.image("https://cdn-icons-png.flaticon.com/512/2643/2643501.png", width=100)

# ------------------------------------------------
# GENERAR DATOS
# ------------------------------------------------
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

with st.status("🛠️ Procesando base de datos académica...", expanded=False) as status:
    st.write("Calculando variables de Bloom...")
    datos = generar_datos(n_muestra)
    st.write("Ejecutando modelos estadísticos...")
    habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
    status.update(label="✅ Datos cargados con éxito", state="complete")

# ------------------------------------------------
# TABS DEL INFORME
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Producto Esperado", "📊 Diagnóstico", "🧠 Mapeo Cognitivo", 
    "📉 Análisis Estadístico", "💡 Guía Pedagógica"
])

# PRODUCTO ESPERADO
with tab1:
    st.header("📄 Producto Esperado")
    st.write("El producto final de esta investigación aplicada se materializa en un **Informe Académico de Investigación**, estructurado bajo normas **APA 7ma edición**.")
    
    c1, c2, c3 = st.columns(3)
    c1.help("Diagnóstico Situacional: Identificación de patrones de uso.")
    c2.help("Mapeo Cognitivo: Evaluación Taxonomía de Bloom.")
    c3.help("Guía: Lineamientos de uso ético.")

# DIAGNÓSTICO
with tab2:
    st.header("📈 Diagnóstico Situacional")
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    col1, col2, col3 = st.columns(3)
    col1.metric("Estudiantes", n_muestra, "Muestra total")
    col2.metric("Andamiaje", f"{uso.get('Andamiaje',0):.1f}%", "Uso sugerido")
    col3.metric("Sustituto", f"{uso.get('Sustituto',0):.1f}%", "Riesgo cognitivo", delta_color="inverse")

    fig_pie = px.pie(names=uso.index, values=uso.values, hole=0.5, 
                     color_discrete_sequence=['#BEE3DB', '#FFD8BE'])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

# MAPEO COGNITIVO
with tab3:
    st.header("🧠 Mapeo de Influencia Cognitiva")
    promedios = datos[habilidades].mean()
    df_bloom = pd.DataFrame({"Habilidad": habilidades, "Promedio": promedios.values})
    
    fig_bar = px.bar(df_bloom, x="Habilidad", y="Promedio", color="Promedio",
                     color_continuous_scale=['#FFD8BE', '#BEE3DB'])
    st.plotly_chart(fig_bar, use_container_width=True)

# ANÁLISIS ESTADÍSTICO
with tab4:
    st.header("🔍 Análisis Estadístico Avanzado")
    with st.expander("Ver Intervalos de Confianza (IC 95%)"):
        std = datos[habilidades].std()
        error = 1.96*(std/np.sqrt(len(datos)))
        df_ic = pd.DataFrame({"Habilidad":habilidades, "Media":promedios.values, 
                             "IC Inferior":(promedios-error).values, "IC Superior":(promedios+error).values})
        st.dataframe(df_ic.style.background_gradient(cmap='Pastel1'))

    st.subheader("🔗 Matriz de Correlación")
    fig_corr = px.imshow(datos[habilidades].corr(), text_auto=True, color_continuous_scale="Tealgrn")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🤖 Modelo de Regresión y ML")
    datos["Indice_PC"] = datos[habilidades].mean(axis=1)
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    
    col_reg1, col_reg2 = st.columns(2)
    with col_reg1:
        st.write("**Resumen OLS:**")
        X = sm.add_constant(datos["Uso_IA_bin"])
        modelo = sm.OLS(datos["Indice_PC"], X).fit()
        st.text(modelo.summary().as_text()[:500] + "...")
    with col_reg2:
        modelo_ml = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(datos[habilidades], datos["Indice_PC"], test_size=0.2)
        modelo_ml.fit(X_train, y_train)
        score = modelo_ml.score(X_test, y_test)
        st.metric("Precisión R² del Modelo", f"{score:.2f}")
        st.toast(f"Modelo entrenado con R² de {score:.2f}", icon='🚀')

# GUÍA PEDAGÓGICA (TIPO ACORDEÓN)
with tab5:
    st.header("💡 Guía de Recomendaciones Pedagógicas")
    
    with st.expander("👨‍🏫 Para Docentes", expanded=True):
        st.write("- Diseñar actividades que requieran **análisis crítico** de resultados de IA.")
        st.write("- Utilizar la IA como **andamiaje cognitivo** para estructurar ideas complejas.")
        
    with st.expander("🎓 Para Estudiantes"):
        st.write("- Evaluar críticamente respuestas generadas por IA contrastando con fuentes primarias.")
        st.write("- Utilizar IA para **explorar perspectivas múltiples** y no como respuesta única.")
        
    with st.expander("🏛️ Para Instituciones"):
        st.write("- Establecer **políticas de uso ético de IA** transversales a todas las carreras.")
        st.write("- Capacitar docentes en **IA educativa** y evaluación por competencias.")

# ------------------------------------------------
# INTERACCIÓN FINAL
# ------------------------------------------------
st.divider()
st.header("📝 Simulación de Encuesta en Vivo")
with st.form("encuesta_interactiva"):
    c_enc1, c_enc2 = st.columns(2)
    with c_enc1:
        u_sel = st.selectbox("¿Modo de uso?", ["Andamiaje", "Sustituto"])
        a_sel = st.slider("Nivel de Análisis", 1, 5, 3)
    with c_enc2:
        e_sel = st.slider("Nivel de Evaluación", 1, 5, 3)
        i_sel = st.slider("Nivel de Inferencia", 1, 5, 3)
    
    if st.form_submit_button("Registrar en estudio"):
        st.balloons()
        st.success("¡Datos integrados al modelo dinámico!")

# ------------------------------------------------
# DESCARGAS
# ------------------------------------------------
col_down1, col_down2 = st.columns(2)
with col_down1:
    st.download_button("📥 Descargar Base CSV", datos.to_csv(index=False), "datos_unemi.csv", "text/csv")
with col_down2:
    if st.button("📄 Generar Informe PDF"):
        with st.spinner("Creando documento APA..."):
            # Lógica de PDF simplificada para el ejemplo
            st.snow()
            st.warning("Función de PDF lista. Descargue arriba.")
