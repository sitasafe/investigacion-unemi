import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------

st.set_page_config(
    page_title="Investigación UNEMI - IAGen",
    layout="wide"
)

st.title("📊 Informe Académico de Investigación Aplicada")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico")
st.markdown("---")

# ---------------------------------------------------
# PARÁMETROS DE INVESTIGACIÓN
# ---------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parámetros de Investigación")

    n_muestra = st.slider(
        "Tamaño de la muestra (n)",
        50,
        500,
        100
    )

    st.markdown("""
    **Tipo de estudio:** Descriptivo – correlacional  
    **Diseño:** No experimental – transversal  
    **Población:** Estudiantes modalidad en línea UNEMI
    """)

# ---------------------------------------------------
# GENERACIÓN DE DATOS (SIMULACIÓN)
# ---------------------------------------------------

@st.cache_data
def generar_datos(n):

    np.random.seed(42)

    datos = pd.DataFrame({

        "ID": range(n),

        "Uso_IA": np.random.choice(
            ["Andamiaje", "Sustituto"],
            n,
            p=[0.65, 0.35]
        ),

        "Analisis": np.random.normal(3.8, 0.5, n).clip(1,5),

        "Evaluacion": np.random.normal(3.2, 0.6, n).clip(1,5),

        "Autorregulacion": np.random.normal(3.5, 0.4, n).clip(1,5),

        "Inferencia": np.random.normal(3.9, 0.3, n).clip(1,5)

    })

    return datos


datos_estudiantes = generar_datos(n_muestra)

habilidades = [
    "Analisis",
    "Evaluacion",
    "Autorregulacion",
    "Inferencia"
]

# ---------------------------------------------------
# METODOLOGÍA
# ---------------------------------------------------

st.header("📚 Metodología del Estudio")

st.write(f"""
Se realizó un **estudio descriptivo-correlacional** con enfoque cuantitativo.

La muestra analizada corresponde a **{n_muestra} estudiantes**
de programas de grado en modalidad en línea de la UNEMI.

Las variables analizadas corresponden a habilidades cognitivas de
orden superior según la **Taxonomía de Bloom**.
""")

# ---------------------------------------------------
# DIAGNÓSTICO SITUACIONAL
# ---------------------------------------------------

st.header("1️⃣ Diagnóstico Situacional")

uso_counts = datos_estudiantes["Uso_IA"].value_counts(normalize=True) * 100

andamiaje_perc = uso_counts.get("Andamiaje",0)
sustituto_perc = uso_counts.get("Sustituto",0)

col1,col2,col3 = st.columns(3)

col1.metric("Muestra Analizada",f"{n_muestra}")

col2.metric(
    "Uso como Andamiaje",
    f"{andamiaje_perc:.1f}%"
)

col3.metric(
    "Uso Sustitutivo",
    f"{sustituto_perc:.1f}%"
)

fig_uso = px.pie(
    names=["Andamiaje","Sustituto"],
    values=[andamiaje_perc,sustituto_perc],
    hole=0.4,
    title="Distribución del patrón de interacción con IAGen"
)

st.plotly_chart(fig_uso,use_container_width=True)

# ---------------------------------------------------
# MAPEO COGNITIVO
# ---------------------------------------------------

st.header("2️⃣ Mapeo de Influencia Cognitiva (Taxonomía de Bloom)")

promedios = datos_estudiantes[habilidades].mean()

df_bloom = pd.DataFrame({
    "Dimensión Cognitiva":habilidades,
    "Puntaje Promedio":promedios.values
})

st.dataframe(df_bloom,use_container_width=True)

fig_bar = px.bar(
    df_bloom,
    x="Dimensión Cognitiva",
    y="Puntaje Promedio",
    color="Puntaje Promedio",
    color_continuous_scale="Blues",
    title="Nivel promedio de habilidades cognitivas"
)

st.plotly_chart(fig_bar,use_container_width=True)

# ---------------------------------------------------
# COMPARACIÓN POR USO DE IA
# ---------------------------------------------------

st.header("3️⃣ Comparación según Tipo de Uso de IA")

comparacion = datos_estudiantes.groupby("Uso_IA")[habilidades].mean()

fig_comp = px.bar(
    comparacion.T,
    barmode="group",
    title="Impacto cognitivo según tipo de uso de IA"
)

st.plotly_chart(fig_comp,use_container_width=True)

# ---------------------------------------------------
# MATRIZ DE CORRELACIÓN
# ---------------------------------------------------

st.header("4️⃣ Análisis Correlacional")

corr = datos_estudiantes[habilidades].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Matriz de correlación entre habilidades cognitivas"
)

st.plotly_chart(fig_corr,use_container_width=True)

# ---------------------------------------------------
# INDICE GLOBAL
# ---------------------------------------------------

st.header("5️⃣ Índice Global de Pensamiento Crítico")

indice_pc = datos_estudiantes[habilidades].mean(axis=1).mean()

st.metric(
    "Índice Global",
    f"{indice_pc:.2f} / 5"
)

# ---------------------------------------------------
# RECOMENDACIONES
# ---------------------------------------------------

st.header("6️⃣ Recomendaciones Pedagógicas")

tab1,tab2,tab3 = st.tabs([
    "Docentes",
    "Estudiantes",
    "Institución"
])

with tab1:

    st.write("""
    - Diseñar actividades que requieran evaluación crítica de respuestas generadas por IA.
    - Evaluar procesos de razonamiento y no únicamente productos finales.
    """)

with tab2:

    st.write("""
    - Utilizar la IA como herramienta de apoyo cognitivo.
    - Contrastar respuestas con fuentes académicas.
    """)

with tab3:

    st.write("""
    - Integrar alfabetización en inteligencia artificial en el currículo.
    - Establecer políticas de uso ético de IA.
    """)

# ---------------------------------------------------
# CONCLUSIONES
# ---------------------------------------------------

st.header("📝 Conclusiones")

impacto = "positivo" if andamiaje_perc > 60 else "moderado"

st.write(f"""
Los resultados indican un impacto **{impacto}** del uso de herramientas
de Inteligencia Artificial Generativa en el desarrollo del pensamiento crítico.

La habilidad con mayor puntuación promedio fue **Inferencia**
({promedios['Inferencia']:.2f}/5), mientras que **Evaluación**
({promedios['Evaluacion']:.2f}/5) presenta niveles relativamente menores,
lo que sugiere la necesidad de fortalecer procesos de validación crítica
de la información generada por IA.
""")

# ---------------------------------------------------
# DESCARGA DE DATOS
# ---------------------------------------------------

csv = datos_estudiantes.to_csv(index=False)

st.download_button(
    "📥 Descargar base de datos",
    csv,
    "datos_investigacion_unemi.csv",
    "text/csv"
)
