import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# ------------------------------------------------
# DICCIONARIO DE TRADUCCIÓN
# ------------------------------------------------
idiomas = {
    "Español": {
        "titulo": "📊 Informe Académico de Investigación",
        "sub": "Impacto de la Inteligencia Artificial Generativa en el Pensamiento Crítico",
        "tutor": "Tutor",
        "equipo": "👥 Equipo de Investigación",
        "config": "⚙️ Configuración",
        "muestra": "Tamaño de la muestra",
        "tabs": ["📂 Producto", "📊 Diagnóstico", "🧠 Mapeo", "📉 Estadística", "💡 Guía"],
        "obj": "Objetivo: Analizar la relación entre IAGen y Pensamiento Crítico en la UNEMI.",
        "m_est": "Estudiantes",
        "m_and": "Andamiaje",
        "m_sus": "Sustituto",
        "regresion": "🤖 Modelo de Regresión",
        "dispersion": "🎯 Gráfico de Correlación",
        "encuesta": "📝 Registro de Nuevo Dato (Simulación)",
        "btn_reg": "Actualizar Reporte",
        "exito": "¡Base de datos actualizada!",
        "descarga": "📥 Descargar Base CSV"
    },
    "Italiano": {
        "titulo": "📊 Rapporto Accademico di Ricerca",
        "sub": "Impatto dell'Intelligenza Artificiale Generativa sul Pensiero Critico",
        "tutor": "Tutore",
        "equipo": "👥 Team di Ricerca",
        "config": "⚙️ Impostazioni",
        "muestra": "Dimensione del campione",
        "tabs": ["📂 Prodotto", "📊 Diagnosi", "🧠 Mappatura", "📉 Statistica", "💡 Guida"],
        "obj": "Obiettivo: Analizzare la relazione tra IAGen e Pensiero Critico in UNEMI.",
        "m_est": "Studenti",
        "m_and": "Impalcatura",
        "m_sus": "Sostituto",
        "regresion": "🤖 Modello di Regressione",
        "dispersion": "🎯 Grafico di Correlazione",
        "encuesta": "📝 Registrazione Nuovi Dati",
        "btn_reg": "Aggiorna Rapporto",
        "exito": "Database aggiornato!",
        "descarga": "📥 Scarica il database CSV"
    }
}

# ------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------
st.set_page_config(page_title="Investigación UNEMI", layout="wide")

with st.sidebar:
    st.header("🌐 Lingua")
    sel_idioma = st.radio("Idioma:", ["Español", "Italiano"], horizontal=True)
    lang = idiomas[sel_idioma]
    n_muestra = st.slider(lang["muestra"], 50, 500, 100)

# Inicializar datos en la sesión para que sean persistentes
if 'df_investigacion' not in st.session_state:
    np.random.seed(42)
    st.session_state.df_investigacion = pd.DataFrame({
        "Uso_IA": np.random.choice(["Andamiaje","Sustituto"], n_muestra, p=[0.65,0.35]),
        "Analisis": np.random.normal(3.8,0.5,n_muestra).clip(1,5),
        "Evaluacion": np.random.normal(3.2,0.6,n_muestra).clip(1,5),
        "Autorregulacion": np.random.normal(3.5,0.4,n_muestra).clip(1,5),
        "Inferencia": np.random.normal(3.9,0.3,n_muestra).clip(1,5)
    })

datos = st.session_state.df_investigacion
habilidades = ["Analisis", "Evaluacion", "Autorregulacion", "Inferencia"]
datos["Indice_PC"] = datos[habilidades].mean(axis=1)

# ------------------------------------------------
# DISEÑO (PORTADA)
# ------------------------------------------------
st.title(lang["titulo"])
st.subheader(lang["sub"])
st.write(f"### {lang['equipo']}")
st.divider()

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(lang["tabs"])

with tab2:
    st.header(lang["tabs"][1])
    uso = datos["Uso_IA"].value_counts(normalize=True)*100
    c1, c2, c3 = st.columns(3)
    c1.metric(lang["m_est"], len(datos))
    c2.metric(lang["m_and"], f"{uso.get('Andamiaje',0):.1f}%")
    c3.metric(lang["m_sus"], f"{uso.get('Sustituto',0):.1f}%")
    st.plotly_chart(px.pie(names=uso.index, values=uso.values, hole=0.4, color_discrete_sequence=['#BEE3DB', '#FFD8BE']), use_container_width=True)

with tab4:
    st.header(lang["tabs"][3])
    # Tabla de IC con efecto visual
    promedios = datos[habilidades].mean()
    std = datos[habilidades].std()
    error = 1.96*(std/np.sqrt(len(datos)))
    df_ic = pd.DataFrame({"Habilidad": habilidades, "Media": promedios.values, "IC Inferior": promedios.values - error.values, "IC Superior": promedios.values + error.values})
    st.dataframe(df_ic.style.format(precision=4).bar(subset=['Media'], color='#BEE3DB').highlight_max(subset=['Media'], color='#FFD8BE'), use_container_width=True)
    
    # Regresión
    datos["Uso_IA_bin"] = datos["Uso_IA"].map({"Andamiaje":1, "Sustituto":0})
    X = sm.add_constant(datos["Uso_IA_bin"])
    modelo = sm.OLS(datos["Indice_PC"], X).fit()
    st.text(f"R²: {modelo.rsquared:.4f}")
    
    # Dispersión
    st.plotly_chart(px.scatter(datos, x="Analisis", y="Indice_PC", color="Uso_IA", trendline="ols", color_discrete_sequence=['#BEE3DB', '#FFD8BE']), use_container_width=True)

# ------------------------------------------------
# FORMULARIO ACTIVO (Aquí está la magia)
# ------------------------------------------------
st.divider()
st.header(lang["encuesta"])
with st.form("encuesta_activa"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        u_sel = st.selectbox(lang["m_and"] if sel_idioma=="Italiano" else "Seleccione Modo de Uso", ["Andamiaje", "Sustituto"])
    with col_f2:
        nota_simulada = st.slider("Puntaje Cognitivo (Simulado)", 1.0, 5.0, 4.0)
    
    if st.form_submit_button(lang["btn_reg"]):
        # Crear nueva fila
        nueva_fila = pd.DataFrame({
            "Uso_IA": [u_sel],
            "Analisis": [nota_simulada],
            "Evaluacion": [nota_simulada - 0.2],
            "Autorregulacion": [nota_simulada - 0.1],
            "Inferencia": [nota_simulada + 0.1]
        })
        # Concatenar a los datos existentes en la sesión
        st.session_state.df_investigacion = pd.concat([st.session_state.df_investigacion, nueva_fila], ignore_index=True)
        st.success(lang["exito"])
        st.balloons()
        st.rerun() # Esto hace que todo el reporte se actualice con el nuevo dato

st.download_button(lang["descarga"], datos.to_csv(index=False), "datos_unemi.csv", "text/csv")
