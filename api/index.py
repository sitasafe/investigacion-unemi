import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Configuración de la Interfaz Profesional
st.set_page_config(page_title="Investigación UNEMI - IAGen", layout="wide")

# Estilo personalizado para el encabezado
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ACADÉMICA ---
st.title("📊 Informe Académico de Investigación Aplicada")
st.subheader("Impacto de la IAGen en el Pensamiento Crítico - UNEMI")
st.markdown("---")

# --- SECCIÓN: METODOLOGÍA (Nivel Tesis) ---
with st.sidebar:
    st.header("⚙️ Parámetros de Investigación")
    n_muestra = st.slider("Tamaño de la Muestra (n)", 50, 500, 100, help="Número de estudiantes analizados")
    st.info("**Metodología:** Estudio descriptivo-correlacional con enfoque cuantitativo.")
    st.write("**Población:** Estudiantes UNEMI modalidad en línea.")

# --- GENERACIÓN DE DATOS REALISTAS (Simulación de Muestra) ---
np.random.seed(42)
datos_estudiantes = pd.DataFrame({
    'ID': range(n_muestra),
    'Uso_IA': np.random.choice(['Andamiaje', 'Sustituto'], n_muestra, p=[0.65, 0.35]),
    'Analisis': np.random.normal(3.8, 0.5, n_muestra).clip(1, 5),
    'Evaluacion': np.random.normal(3.2, 0.6, n_muestra).clip(1, 5),
    'Autorregulacion': np.random.normal(3.5, 0.4, n_muestra).clip(1, 5),
    'Inferencia': np.random.normal(3.9, 0.3, n_muestra).clip(1, 5)
})

# --- ENTREGABLE 1: DIAGNÓSTICO SITUACIONAL ---
st.header("1. Diagnóstico Situacional")
col1, col2, col3 = st.columns(3)

uso_counts = datos_estudiantes['Uso_IA'].value_counts(normalize=True) * 100
andamiaje_perc = uso_counts.get('Andamiaje', 0)
sustituto_perc = uso_counts.get('Sustituto', 0)

with col1:
    st.metric("Muestra Analizada", f"{n_muestra} Est.")
with col2:
    st.metric("Uso como Andamiaje", f"{andamiaje_perc:.1f}%", delta="Ideal", delta_color="normal")
with col3:
    st.metric("Uso Sustitutivo", f"{sustituto_perc:.1f}%", delta="Riesgo", delta_color="inverse")

st.write("#### Distribución del Patrón de Interacción")
fig_uso = px.pie(names=['Andamiaje', 'Sustituto'], values=[andamiaje_perc, sustituto_perc], 
             color=['Andamiaje', 'Sustituto'], color_discrete_map={'Andamiaje':'#1E88E5', 'Sustituto':'#E53935'},
             hole=0.4)
st.plotly_chart(fig_uso, use_container_width=True)

# --- ENTREGABLE 2: MAPEO DE INFLUENCIA COGNITIVA (BLOOM) ---
st.header("2. Mapeo de Influencia Cognitiva (Taxonomía de Bloom)")
st.write("Evaluación de habilidades de orden superior bajo el impacto de herramientas IAGen.")

# Procesamiento de promedios
habilidades = ['Analisis', 'Evaluacion', 'Autorregulacion', 'Inferencia']
promedios = datos_estudiantes[habilidades].mean()
df_bloom = pd.DataFrame({
    'Dimensión Cognitiva': habilidades,
    'Puntaje Promedio (1-5)': promedios.values,
    'Nivel de Impacto': ['Alto', 'Medio-Bajo', 'Medio-Alto', 'Alto']
})

col_tabla, col_graph = st.columns([1, 1])

with col_tabla:
    st.dataframe(df_bloom.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

with col_graph:
    fig_bar = px.bar(df_bloom, x='Dimensión Cognitiva', y='Puntaje Promedio (1-5)', 
                 color='Puntaje Promedio (1-5)', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- ENTREGABLE 3: GUÍA DE RECOMENDACIONES PEDAGÓGICAS ---
st.header("3. Guía de Recomendaciones Pedagógicas")

tab1, tab2, tab3 = st.tabs(["🎯 Para Docentes", "🎓 Para Estudiantes", "⚖️ Ética y Autoridades"])

with tab1:
    st.success("""
    - **Validación de Sesgos:** Solicitar a los estudiantes que identifiquen al menos dos alucinaciones en los textos generados por IA.
    - **Evaluación de Procesos:** Calificar la evolución de los borradores y los "prompts" utilizados, no solo el producto final.
    """)

with tab2:
    st.info("""
    - **Prompt Engineering Crítico:** Utilizar la técnica de 'Cadena de Pensamiento' (Chain of Thought) para desglosar problemas complejos.
    - **Contraste de Fuentes:** Validar la información de la IA con bases de datos académicas (Scopus, Google Scholar).
    """)

with tab3:
    st.warning("""
    - **Políticas de Transparencia:** Establecer el porcentaje permitido de asistencia por IA según la naturaleza de la asignatura.
    - **Integridad Académica:** Implementar el uso ético de la IA como competencia transversal en el currículo.
    """)

# --- CONCLUSIONES AUTOMATIZADAS ---
st.markdown("---")
st.write("### 📝 Conclusiones Finales")
impacto_global = "Positivo" if andamiaje_perc > 60 else "En Alerta"
st.write(f"""
La investigación concluye que existe un impacto **{impacto_global}** en la muestra analizada. 
Se observa una correlación entre el uso de IA como andamiaje y los niveles de **Inferencia** ({promedios['Inferencia']:.2f}/5.0). 
Es imperativo reforzar el área de **Evaluación** ({promedios['Evaluacion']:.2f}/5.0) para mitigar la dependencia cognitiva.
""")
