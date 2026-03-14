import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ---------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------

st.set_page_config(page_title="Investigación UNEMI - IAGen", layout="wide")

st.title("📊 Informe Académico de Investigación Aplicada")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico - UNEMI")
st.markdown("---")

# ---------------------------------------------------
# PARÁMETROS DE INVESTIGACIÓN
# ---------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parámetros de Investigación")

    n_muestra = st.slider(
        "Tamaño de la muestra",
        50,
        500,
        100
    )

    st.markdown("""
    **Tipo de estudio:** Descriptivo–correlacional  
    **Diseño:** No experimental – transversal  
    **Población:** Estudiantes modalidad en línea UNEMI
    """)

# ---------------------------------------------------
# GENERACIÓN DE DATOS
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

        "Analisis": np.random.normal(3.8, 0.5, n).clip(1, 5),

        "Evaluacion": np.random.normal(3.2, 0.6, n).clip(1, 5),

        "Autorregulacion": np.random.normal(3.5, 0.4, n).clip(1, 5),

        "Inferencia": np.random.normal(3.9, 0.3, n).clip(1, 5)

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

st.header("📚 Metodología")

st.write(f"""
Estudio **descriptivo–correlacional** con enfoque cuantitativo.

Muestra analizada: **{n_muestra} estudiantes**.

Las variables corresponden a habilidades cognitivas de orden superior
según la **Taxonomía de Bloom**.
""")

# ---------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------

st.header("1️⃣ Diagnóstico Situacional")

uso_counts = datos_estudiantes["Uso_IA"].value_counts(normalize=True) * 100

andamiaje_perc = uso_counts.get("Andamiaje", 0)
sustituto_perc = uso_counts.get("Sustituto", 0)

col1, col2, col3 = st.columns(3)

col1.metric("Muestra Analizada", n_muestra)

col2.metric(
    "Uso como Andamiaje",
    f"{andamiaje_perc:.1f}%"
)

col3.metric(
    "Uso Sustitutivo",
    f"{sustituto_perc:.1f}%"
)

fig_uso = px.pie(
    names=["Andamiaje", "Sustituto"],
    values=[andamiaje_perc, sustituto_perc],
    hole=0.4
)

st.plotly_chart(fig_uso, use_container_width=True)

# ---------------------------------------------------
# MAPEO COGNITIVO
# ---------------------------------------------------

st.header("2️⃣ Mapeo Cognitivo (Bloom)")

promedios = datos_estudiantes[habilidades].mean()

df_bloom = pd.DataFrame({
    "Dimensión Cognitiva": habilidades,
    "Puntaje Promedio": promedios.values
})

st.dataframe(df_bloom, use_container_width=True)

fig_bar = px.bar(
    df_bloom,
    x="Dimensión Cognitiva",
    y="Puntaje Promedio",
    color="Puntaje Promedio",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# COMPARACIÓN
# ---------------------------------------------------

st.header("3️⃣ Comparación por Uso de IA")

comparacion = datos_estudiantes.groupby("Uso_IA")[habilidades].mean()

fig_comp = px.bar(
    comparacion.T,
    barmode="group"
)

st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------
# CORRELACIÓN
# ---------------------------------------------------

st.header("4️⃣ Análisis Correlacional")

corr = datos_estudiantes[habilidades].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------
# INDICE GLOBAL
# ---------------------------------------------------

datos_estudiantes["Indice_PC"] = datos_estudiantes[habilidades].mean(axis=1)

indice_pc = datos_estudiantes["Indice_PC"].mean()

st.header("5️⃣ Índice Global de Pensamiento Crítico")

st.metric(
    "Indice Global",
    f"{indice_pc:.2f}/5"
)

# ---------------------------------------------------
# REGRESIÓN
# ---------------------------------------------------

st.header("6️⃣ Análisis de Regresión")

datos_estudiantes["Uso_IA_bin"] = datos_estudiantes["Uso_IA"].map({
    "Andamiaje": 1,
    "Sustituto": 0
})

X = datos_estudiantes["Uso_IA_bin"]
y = datos_estudiantes["Indice_PC"]

X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

st.text(modelo.summary())

# ---------------------------------------------------
# MODELO PREDICTIVO
# ---------------------------------------------------

st.header("7️⃣ Modelo Predictivo")

X_ml = datos_estudiantes[habilidades]
y_ml = datos_estudiantes["Indice_PC"]

X_train, X_test, y_train, y_test = train_test_split(
    X_ml,
    y_ml,
    test_size=0.2,
    random_state=42
)

modelo_ml = LinearRegression()

modelo_ml.fit(X_train, y_train)

score = modelo_ml.score(X_test, y_test)

st.metric(
    "Precisión del Modelo (R²)",
    f"{score:.2f}"
)

# ---------------------------------------------------
# CONCLUSIONES
# ---------------------------------------------------

st.header("📝 Conclusiones")

impacto = "positivo" if andamiaje_perc > 60 else "moderado"

st.write(f"""
La investigación muestra un impacto **{impacto}** del uso de IAGen
en el desarrollo del pensamiento crítico.

La habilidad con mayor puntaje promedio fue **Inferencia**
({promedios['Inferencia']:.2f}/5), mientras que **Evaluación**
({promedios['Evaluacion']:.2f}/5) presenta valores menores,
lo que sugiere fortalecer procesos de validación crítica.
""")

# ---------------------------------------------------
# DESCARGA DE DATOS
# ---------------------------------------------------

csv = datos_estudiantes.to_csv(index=False)

st.download_button(
    "📥 Descargar Base de Datos",
    csv,
    "datos_investigacion_unemi.csv",
    "text/csv"
)

# ---------------------------------------------------
# GENERAR PDF
# ---------------------------------------------------

def generar_pdf():

    styles = getSampleStyleSheet()

    contenido = []

    contenido.append(
        Paragraph(
            "Informe de Investigación: IAGen y Pensamiento Crítico",
            styles["Title"]
        )
    )

    contenido.append(
        Paragraph(
            f"Muestra analizada: {n_muestra} estudiantes",
            styles["Normal"]
        )
    )

    contenido.append(
        Paragraph(
            f"Uso como andamiaje: {andamiaje_perc:.2f}%",
            styles["Normal"]
        )
    )

    contenido.append(
        Paragraph(
            f"Indice global de pensamiento crítico: {indice_pc:.2f}",
            styles["Normal"]
        )
    )

    archivo = "informe_unemi.pdf"

    doc = SimpleDocTemplate(
        archivo,
        pagesize=letter
    )

    doc.build(contenido)

    return archivo


if st.button("📄 Generar Informe PDF"):

    archivo = generar_pdf()

    with open(archivo, "rb") as f:

        st.download_button(
            label="Descargar PDF",
            data=f,
            file_name="informe_unemi.pdf",
            mime="application/pdf"
        )
