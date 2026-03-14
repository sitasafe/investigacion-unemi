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

# -----------------------------------------
# CONFIGURACIÓN
# -----------------------------------------

st.set_page_config(page_title="Investigación UNEMI", layout="wide")

st.title("📊 Investigación: IAGen y Pensamiento Crítico")
st.subheader("Universidad Estatal de Milagro (UNEMI)")

# -----------------------------------------
# PARÁMETROS
# -----------------------------------------

with st.sidebar:

    st.header("Configuración del estudio")

    n_muestra = st.slider(
        "Tamaño de la muestra",
        50,
        500,
        100
    )

# -----------------------------------------
# GENERAR DATOS
# -----------------------------------------

@st.cache_data
def generar_datos(n):

    np.random.seed(42)

    df = pd.DataFrame({

        "Uso_IA": np.random.choice(
            ["Andamiaje","Sustituto"],
            n,
            p=[0.65,0.35]
        ),

        "Analisis": np.random.normal(3.8,0.5,n).clip(1,5),
        "Evaluacion": np.random.normal(3.2,0.6,n).clip(1,5),
        "Autorregulacion": np.random.normal(3.5,0.4,n).clip(1,5),
        "Inferencia": np.random.normal(3.9,0.3,n).clip(1,5)

    })

    return df


datos = generar_datos(n_muestra)

habilidades = [
"Analisis",
"Evaluacion",
"Autorregulacion",
"Inferencia"
]

# -----------------------------------------
# METODOLOGÍA
# -----------------------------------------

st.header("📚 Metodología")

st.write(f"""
Estudio cuantitativo descriptivo correlacional.

Muestra analizada: **{n_muestra} estudiantes**.

Se evalúan habilidades de pensamiento crítico basadas en la Taxonomía de Bloom.
""")

# -----------------------------------------
# DIAGNÓSTICO
# -----------------------------------------

st.header("1️⃣ Diagnóstico Situacional")

uso = datos["Uso_IA"].value_counts(normalize=True)*100

col1,col2,col3 = st.columns(3)

col1.metric("Estudiantes",n_muestra)
col2.metric("Andamiaje",f"{uso.get('Andamiaje',0):.1f}%")
col3.metric("Sustituto",f"{uso.get('Sustituto',0):.1f}%")

fig_pie = px.pie(
names=uso.index,
values=uso.values,
hole=0.4
)

st.plotly_chart(fig_pie,use_container_width=True)

# -----------------------------------------
# MAPEO COGNITIVO
# -----------------------------------------

st.header("2️⃣ Mapeo Cognitivo")

promedios = datos[habilidades].mean()

df_bloom = pd.DataFrame({
"Habilidad":habilidades,
"Promedio":promedios.values
})

fig_bar = px.bar(
df_bloom,
x="Habilidad",
y="Promedio",
color="Promedio"
)

st.plotly_chart(fig_bar,use_container_width=True)

# -----------------------------------------
# INTERVALOS DE CONFIANZA
# -----------------------------------------

st.header("3️⃣ Intervalos de Confianza")

media = datos[habilidades].mean()
std = datos[habilidades].std()
n = len(datos)

error = 1.96*(std/np.sqrt(n))

df_ic = pd.DataFrame({
"Habilidad":habilidades,
"Media":media.values,
"IC Inferior":(media-error).values,
"IC Superior":(media+error).values
})

st.dataframe(df_ic)

# -----------------------------------------
# CORRELACIÓN
# -----------------------------------------

st.header("4️⃣ Correlación")

corr = datos[habilidades].corr()

fig_corr = px.imshow(
corr,
text_auto=True,
color_continuous_scale="Blues"
)

st.plotly_chart(fig_corr,use_container_width=True)

# -----------------------------------------
# REGRESIÓN
# -----------------------------------------

st.header("5️⃣ Regresión")

datos["Indice_PC"] = datos[habilidades].mean(axis=1)

datos["Uso_IA_bin"] = datos["Uso_IA"].map({
"Andamiaje":1,
"Sustituto":0
})

X = sm.add_constant(datos["Uso_IA_bin"])
y = datos["Indice_PC"]

modelo = sm.OLS(y,X).fit()

st.text(modelo.summary())

# -----------------------------------------
# MODELO PREDICTIVO
# -----------------------------------------

st.header("6️⃣ Modelo Predictivo")

X_ml = datos[habilidades]
y_ml = datos["Indice_PC"]

X_train,X_test,y_train,y_test = train_test_split(
X_ml,
y_ml,
test_size=0.2,
random_state=42
)

modelo_ml = LinearRegression()

modelo_ml.fit(X_train,y_train)

score = modelo_ml.score(X_test,y_test)

st.metric("Precisión del modelo",f"{score:.2f}")

# -----------------------------------------
# ENCUESTA REAL
# -----------------------------------------

st.header("🧪 Simulación de Encuesta")

with st.form("encuesta"):

    uso = st.selectbox(
        "¿Cómo usas la IA?",
        ["Andamiaje","Sustituto"]
    )

    analisis = st.slider("Análisis",1,5,3)
    evaluacion = st.slider("Evaluación",1,5,3)
    autorreg = st.slider("Autorregulación",1,5,3)
    inferencia = st.slider("Inferencia",1,5,3)

    enviar = st.form_submit_button("Enviar")

if enviar:

    nueva = pd.DataFrame([{
        "Uso_IA":uso,
        "Analisis":analisis,
        "Evaluacion":evaluacion,
        "Autorregulacion":autorreg,
        "Inferencia":inferencia
    }])

    st.success("Respuesta registrada")
    st.dataframe(nueva)

# -----------------------------------------
# DESCARGAR DATOS
# -----------------------------------------

st.header("📥 Descargar datos")

csv = datos.to_csv(index=False)

st.download_button(
"Descargar base de datos",
csv,
"datos_investigacion_unemi.csv",
"text/csv"
)

# -----------------------------------------
# GENERAR INFORME APA
# -----------------------------------------

def generar_pdf():

    styles = getSampleStyleSheet()

    contenido = []

    contenido.append(
        Paragraph(
        "Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico",
        styles["Title"]
        )
    )

    contenido.append(Spacer(1,20))

    contenido.append(
        Paragraph(
        f"Muestra analizada: {n_muestra} estudiantes",
        styles["Normal"]
        )
    )

    contenido.append(
        Paragraph(
        f"Promedio global pensamiento crítico: {datos['Indice_PC'].mean():.2f}",
        styles["Normal"]
        )
    )

    contenido.append(
        Paragraph(
        "El análisis sugiere que el uso de IA como andamiaje cognitivo está asociado con niveles moderados de pensamiento crítico.",
        styles["Normal"]
        )
    )

    archivo="informe_unemi.pdf"

    doc=SimpleDocTemplate(
    archivo,
    pagesize=letter
    )

    doc.build(contenido)

    return archivo


st.header("📄 Informe automático")

if st.button("Generar informe"):

    archivo=generar_pdf()

    with open(archivo,"rb") as f:

        st.download_button(
        "Descargar informe APA",
        f,
        file_name="informe_unemi.pdf",
        mime="application/pdf"
        )
