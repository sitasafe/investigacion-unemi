import streamlit as st
import pandas as pd
import numpy as np

# Configuración de página
st.set_page_config(page_title="Investigación UNEMI - IAGen", layout="centered")

# -----------------------------
# SIMULACIÓN DE DATOS
# -----------------------------
@st.cache_data
def generar_datos():
    np.random.seed(42)

    n = 100

    uso = np.random.choice(
        ["Andamiaje Cognitivo", "Sustituto Intelectual"],
        size=n,
        p=[0.65, 0.35]
    )

    habilidades = {
        "Analisis": np.random.normal(3.8, 0.3, n),
        "Evaluacion": np.random.normal(3.2, 0.4, n),
        "Autorregulacion": np.random.normal(3.5, 0.3, n),
        "Inferencia": np.random.normal(3.9, 0.2, n)
    }

    df = pd.DataFrame(habilidades)
    df["Uso_IA"] = uso

    return df

df = generar_datos()

# -----------------------------
# TITULO
# -----------------------------

st.title("Influencia de la Inteligencia Artificial Generativa en el Pensamiento Crítico")

st.subheader("Universidad Estatal de Milagro (UNEMI)")

st.write("""
Este informe presenta los resultados de una **investigación aplicada**
sobre la relación entre el uso de herramientas de **Inteligencia Artificial Generativa (IAGen)** 
y el desarrollo del **pensamiento crítico** en estudiantes de modalidad en línea.
""")

# -----------------------------
# METODOLOGÍA
# -----------------------------

st.write("## Metodología")

st.write("""
**Tipo de estudio:** Investigación aplicada de enfoque cuantitativo.

**Muestra:** 100 estudiantes de grado en modalidad en línea.

**Variables analizadas:**

- Uso de Inteligencia Artificial Generativa
- Habilidades cognitivas de orden superior según la **Taxonomía de Bloom**
""")

# -----------------------------
# DIAGNÓSTICO SITUACIONAL
# -----------------------------

st.write("## Diagnóstico Situacional")

uso_counts = df["Uso_IA"].value_counts()

porcentaje_andamiaje = (uso_counts["Andamiaje Cognitivo"] / len(df)) * 100
porcentaje_sustituto = (uso_counts["Sustituto Intelectual"] / len(df)) * 100

col1, col2 = st.columns(2)

col1.metric("Uso como Andamiaje Cognitivo", f"{porcentaje_andamiaje:.1f}%")
col2.metric("Uso como Sustituto Intelectual", f"{porcentaje_sustituto:.1f}%")

st.bar_chart(uso_counts)

st.info("""
El análisis evidencia que la mayoría de los estudiantes utilizan la IAGen
como **herramienta de apoyo cognitivo**, mientras que un porcentaje menor
presenta patrones de dependencia tecnológica.
""")

# -----------------------------
# MAPEO COGNITIVO (BLOOM)
# -----------------------------

st.write("## Mapeo de Influencia Cognitiva")

promedios = df.drop(columns=["Uso_IA"]).mean()

tabla = pd.DataFrame({
    "Habilidad Cognitiva": promedios.index,
    "Nivel Promedio": promedios.values
})

st.dataframe(tabla)

st.bar_chart(promedios)

st.write("""
Las habilidades evaluadas corresponden a niveles superiores de la **Taxonomía de Bloom**:

- **Análisis**
- **Evaluación**
- **Autorregulación**
- **Inferencia**
""")

# -----------------------------
# RECOMENDACIONES
# -----------------------------

st.write("## Guía de Recomendaciones Pedagógicas")

st.success("""
**Para docentes**

- Diseñar actividades que requieran validación crítica de respuestas generadas por IA.
- Promover ejercicios de comparación entre respuestas humanas y generadas por algoritmos.

**Para estudiantes**

- Utilizar técnicas de *Prompt Engineering* orientadas al auto-cuestionamiento.
- Emplear la IA como apoyo para la reflexión, no como sustituto del razonamiento.

**Para instituciones**

- Establecer políticas de uso ético de la inteligencia artificial.
- Integrar la alfabetización en IA dentro del currículo universitario.
""")

# -----------------------------
# CONCLUSIONES
# -----------------------------

st.write("## Conclusiones")

st.write(f"""
El estudio evidencia que aproximadamente **{porcentaje_andamiaje:.1f}%**
de los estudiantes utilizan la inteligencia artificial como **herramienta
de apoyo cognitivo**, lo cual sugiere que estas tecnologías pueden actuar
como **catalizadores del pensamiento crítico** cuando se integran de forma
pedagógicamente adecuada.

Sin embargo, el **{porcentaje_sustituto:.1f}%** restante evidencia riesgos
de dependencia tecnológica, lo que plantea la necesidad de fortalecer
estrategias de **uso crítico y ético de la IA en la educación superior**.
""")
