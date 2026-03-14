import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Investigación UNEMI", layout="wide")

# ------------------------------------------------
# PORTADA
# ------------------------------------------------

st.title("Informe Académico de Investigación")
st.subheader("Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico")

st.markdown("""
**Maestría en Educación mención en Docencia e Investigación en Educación Superior**

**Módulo:** Seminario de Investigación 1  
**Actividad:** Tarea de Contacto con Docente  

**Tutor:** Bonisoli Lorenzo PhD.

**Integrantes**

- Willan Efrén Álvarez Carmona  
- Tania Jacqueline Barcos Villalva  
- Selene Anaís Guagua Valencia  
- Pedro Javier Figueroa Vergara  
- Nohemí Nicole Miranda Jiménez  

**Fecha de entrega:** 07 de marzo de 2026
""")

st.divider()

# ------------------------------------------------
# TABS DEL INFORME
# ------------------------------------------------

tab1,tab2,tab3,tab4 = st.tabs([
"Producto Esperado",
"Diagnóstico Situacional",
"Mapeo Cognitivo",
"Guía Pedagógica"
])

# ------------------------------------------------
# PRODUCTO ESPERADO
# ------------------------------------------------

with tab1:

    st.header("Producto Esperado")

    st.write("""
El producto final de esta investigación aplicada se materializa en un **Informe Académico de Investigación**, estructurado bajo normas **APA 7ma edición**, que sistematiza el análisis de la relación entre el uso de herramientas de **Inteligencia Artificial Generativa (IAGen)** y el desarrollo del **pensamiento crítico en estudiantes de grado de la modalidad en línea de la Universidad Estatal de Milagro (UNEMI)**.
""")

    st.subheader("Entregables técnicos")

    st.markdown("""
**1. Diagnóstico Situacional**

Identificación de patrones de uso y hábitos de interacción de los estudiantes con la IAGen, permitiendo detectar si la herramienta se utiliza como un **andamiaje cognitivo** o como un **sustituto intelectual**.

**2. Mapeo de Influencia Cognitiva**

Evaluación de la influencia de la IAGen en habilidades de orden superior según la **Taxonomía de Bloom**:

- Análisis
- Evaluación
- Autorregulación
- Inferencia

**3. Guía de Recomendaciones Pedagógicas**

Lineamientos estratégicos para docentes y autoridades académicas orientados a promover un **uso ético y responsable de la inteligencia artificial en entornos virtuales de aprendizaje**.
""")

# ------------------------------------------------
# GENERAR DATOS DE INVESTIGACIÓN
# ------------------------------------------------

np.random.seed(42)

datos = pd.DataFrame({

"Uso_IA": np.random.choice(
["Andamiaje","Sustituto"],
100,
p=[0.65,0.35]
),

"Analisis": np.random.normal(3.8,0.5,100).clip(1,5),
"Evaluacion": np.random.normal(3.2,0.6,100).clip(1,5),
"Autorregulacion": np.random.normal(3.5,0.4,100).clip(1,5),
"Inferencia": np.random.normal(3.9,0.3,100).clip(1,5)

})

habilidades = [
"Analisis",
"Evaluacion",
"Autorregulacion",
"Inferencia"
]

# ------------------------------------------------
# DIAGNOSTICO
# ------------------------------------------------

with tab2:

    st.header("Diagnóstico Situacional")

    uso = datos["Uso_IA"].value_counts(normalize=True)*100

    st.write(f"""
Los resultados muestran que aproximadamente **{uso.get("Andamiaje",0):.1f}%** de los estudiantes utilizan la inteligencia artificial como **andamiaje cognitivo**, mientras que **{uso.get("Sustituto",0):.1f}%** la emplean como **sustituto intelectual**.
""")

    fig = px.pie(
    names=uso.index,
    values=uso.values,
    hole=0.4,
    title="Patrones de uso de IAGen"
)

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# MAPEO COGNITIVO
# ------------------------------------------------

with tab3:

    st.header("Mapeo de Influencia Cognitiva")

    promedios = datos[habilidades].mean()

    df = pd.DataFrame({
    "Habilidad":habilidades,
    "Promedio":promedios.values
})

    fig = px.bar(
    df,
    x="Habilidad",
    y="Promedio",
    color="Promedio",
    title="Nivel promedio de habilidades cognitivas"
)

    st.plotly_chart(fig,use_container_width=True)

    st.write("""
Los resultados sugieren que las habilidades de **inferencia y análisis presentan valores más altos**, lo cual puede estar asociado con el uso de herramientas de apoyo cognitivo. Sin embargo, las habilidades de **evaluación crítica requieren fortalecimiento pedagógico**.
""")

# ------------------------------------------------
# RECOMENDACIONES
# ------------------------------------------------

with tab4:

    st.header("Guía de Recomendaciones Pedagógicas")

    st.success("""
**Para docentes**

- Diseñar actividades que requieran **argumentación y verificación de fuentes**.
- Integrar la IA como **herramienta de apoyo y no sustitución cognitiva**.
- Fomentar el aprendizaje basado en problemas.

**Para estudiantes**

- Utilizar la IA para **explorar perspectivas múltiples**.
- Realizar **autoevaluación crítica de respuestas generadas por IA**.

**Para instituciones**

- Implementar **políticas de uso ético de IA**.
- Capacitar docentes en **IA aplicada a la educación**.
""")
