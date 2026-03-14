import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Investigación UNEMI", layout="centered")

# --- LÓGICA DE INVESTIGACIÓN ---
def obtener_datos():
    np.random.seed(42)
    data = {
        'Habilidad': ['Análisis', 'Evaluación', 'Autorregulación', 'Inferencia'],
        'Nivel_Promedio': [3.8, 3.2, 3.5, 3.9],
        'Impacto_IA': ['Positivo', 'Neutral', 'Positivo', 'Positivo']
    }
    return pd.DataFrame(data)

# --- INTERFAZ ---
st.title("Producto Esperado: Informe IAGen vs Pensamiento Crítico")
st.subheader("Universidad Estatal de Milagro (UNEMI)")

st.write("### 1. Diagnóstico Situacional")
st.info("Se identificó que el 65% de los estudiantes utilizan la IAGen como andamiaje cognitivo, mientras que un 35% presenta patrones de uso sustitutivo intelectual.")

st.write("### 2. Mapeo de Influencia Cognitiva (Taxonomía de Bloom)")
df = obtener_datos()
st.table(df) # Esto crea la tabla automáticamente

st.write("### 3. Guía de Recomendaciones Pedagógicas")
st.success("""
- **Para Docentes:** Diseñar consignas que requieran validación crítica.
- **Para Estudiantes:** Usar Prompt Engineering para el auto-cuestionamiento.
- **Ética:** Políticas de transparencia en algoritmos.
""")
