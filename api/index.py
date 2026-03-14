import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------

st.set_page_config(page_title="Investigación UNEMI - IAGen", layout="wide")

st.title("📊 Informe Académico Interactivo")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico")

# ------------------------------------------------
# PARÁMETROS
# ------------------------------------------------

with st.sidebar:

    st.header("Parámetros de investigación")

    n_muestra = st.slider(
        "Tamaño de la muestra",
        50,
        500,
        100
    )

# ------------------------------------------------
# GENERAR DATOS
# ------------------------------------------------

@st.cache_data
def generar_datos(n):

    np.random.seed(42)

    datos = pd.DataFrame({

        "ID": range(n),

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

    return datos


datos_estudiantes = generar_datos(n_muestra)

habilidades = [
"Analisis",
"Evaluacion",
"Autorregulacion",
"Inferencia"
]

# ------------------------------------------------
# METODOLOGÍA
# ------------------------------------------------

st.header("📚 Metodología")

st.write(f"""
Estudio **descriptivo correlacional** con enfoque cuantitativo.

Muestra analizada: **{n_muestra} estudiantes**.

Variables analizadas basadas en la **Taxonomía de Bloom**.
""")

# ------------------------------------------------
# DIAGNÓSTICO
# ------------------------------------------------

st.header("1️⃣ Diagnóstico Situacional")

uso_counts = datos_estudiantes["Uso_IA"].value_counts(normalize=True)*100

andamiaje_perc = uso_counts.get("Andamiaje",0)
sustituto_perc = uso_counts.get("Sustituto",0)

col1,col2,col3 = st.columns(3)

col1.metric("Muestra",n_muestra)
col2.metric("Uso Andamiaje",f"{andamiaje_perc:.1f}%")
col3.metric("Uso Sustituto",f"{sustituto_perc:.1f}%")

fig_uso = px.pie(
names=["Andamiaje","Sustituto"],
values=[andamiaje_perc,sustituto_perc],
hole=0.4
)

st.plotly_chart(fig_uso,use_container_width=True)

# ------------------------------------------------
# MAPEO COGNITIVO
# ------------------------------------------------

st.header("2️⃣ Mapeo Cognitivo")

promedios = datos_estudiantes[habilidades].mean()

df_bloom = pd.DataFrame({
"Dimensión":habilidades,
"Promedio":promedios.values
})

fig_bar = px.bar(
df_bloom,
x="Dimensión",
y="Promedio",
color="Promedio",
color_continuous_scale="Blues"
)

st.plotly_chart(fig_bar,use_container_width=True)

# ------------------------------------------------
# INTERVALOS DE CONFIANZA
# ------------------------------------------------

st.header("3️⃣ Intervalos de Confianza")

media = datos_estudiantes[habilidades].mean()
std = datos_estudiantes[habilidades].std()
n = len(datos_estudiantes)

error = 1.96*(std/np.sqrt(n))

df_ic = pd.DataFrame({
"Habilidad":habilidades,
"Media":media.values,
"IC_inf":(media-error).values,
"IC_sup":(media+error).values
})

st.dataframe(df_ic)

# ------------------------------------------------
# CORRELACIÓN
# ------------------------------------------------

st.header("4️⃣ Correlación")

corr = datos_estudiantes[habilidades].corr()

fig_corr = px.imshow(
corr,
text_auto=True,
color_continuous_scale="Blues"
)

st.plotly_chart(fig_corr,use_container_width=True)

# ------------------------------------------------
# REGRESIÓN
# ------------------------------------------------

st.header("5️⃣ Regresión")

datos_estudiantes["Indice_PC"]=datos_estudiantes[habilidades].mean(axis=1)

datos_estudiantes["Uso_IA_bin"]=datos_estudiantes["Uso_IA"].map({
"Andamiaje":1,
"Sustituto":0
})

X=datos_estudiantes["Uso_IA_bin"]
y=datos_estudiantes["Indice_PC"]

X=sm.add_constant(X)

modelo=sm.OLS(y,X).fit()

st.text(modelo.summary())

# ------------------------------------------------
# MODELO PREDICTIVO
# ------------------------------------------------

st.header("6️⃣ Modelo Predictivo")

X_ml=datos_estudiantes[habilidades]
y_ml=datos_estudiantes["Indice_PC"]

X_train,X_test,y_train,y_test=train_test_split(
X_ml,
y_ml,
test_size=0.2,
random_state=42
)

modelo_ml=LinearRegression()

modelo_ml.fit(X_train,y_train)

score=modelo_ml.score(X_test,y_test)

st.metric("Precisión del modelo",f"{score:.2f}")

# ------------------------------------------------
# ENCUESTA
# ------------------------------------------------

st.header("🧪 Encuesta de estudiantes")

with st.form("encuesta"):

    uso=st.selectbox("Uso de IA",["Andamiaje","Sustituto"])

    analisis=st.slider("Analisis",1,5,3)
    evaluacion=st.slider("Evaluacion",1,5,3)
    autorreg=st.slider("Autorregulación",1,5,3)
    inferencia=st.slider("Inferencia",1,5,3)

    enviar=st.form_submit_button("Enviar")

if enviar:

    st.success("Respuesta registrada.")

# ------------------------------------------------
# DESCARGAR DATOS
# ------------------------------------------------

csv=datos_estudiantes.to_csv(index=False)

st.download_button(
"Descargar base de datos",
csv,
"datos_unemi.csv",
"text/csv"
)

# ------------------------------------------------
# GENERAR INFORME APA
# ------------------------------------------------

def generar_pdf():

    styles=getSampleStyleSheet()

    contenido=[]

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
        f"Uso de IA como andamiaje: {andamiaje_perc:.2f}%",
        styles["Normal"]
        )
    )

    contenido.append(
        Paragraph(
        f"Indice global de pensamiento crítico: {datos_estudiantes['Indice_PC'].mean():.2f}",
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

if st.button("📄 Generar informe APA"):

    archivo=generar_pdf()

    with open(archivo,"rb") as f:

        st.download_button(
        "Descargar informe",
        f,
        file_name="informe_unemi.pdf",
        mime="application/pdf"
        )
