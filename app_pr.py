# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"  # No modificar

SYSTEM_PROMPT = """Eres un asistente analítico de Spotify. Un usuario quiere explorar sus hábitos de escucha.

DATOS DISPONIBLES:
Tienes acceso a un DataFrame llamado `df` con las siguientes columnas:
- ts (datetime): timestamp de la reproducción
- ms_played (int): milisegundos de reproducción
- min_played (float): minutos de reproducción (= ms_played / 60000)
- track_name (str): nombre de la canción (null si es podcast)
- artist_name (str): nombre del artista (null si es podcast)
- album_name (str): nombre del álbum (null si es podcast)
- reason_start (str): motivo de inicio ({reason_start_values})
- reason_end (str): motivo de fin ({reason_end_values})
- shuffle (bool): si estaba en modo aleatorio
- skipped (bool o null): True si se saltó, null si no se saltó
- platform (str): plataforma ({plataformas})
- hour (int): hora del día (0-23)
- weekday (str): día de la semana (Monday, Tuesday, ..., Sunday)
- weekday_num (int): día de la semana como número (0=Monday, 6=Sunday)
- month (int): mes (1-12)
- month_name (str): nombre del mes (January, February, ..., December)
- is_weekend (bool): True si es sábado o domingo
- is_podcast (bool): True si es un podcast (track_name es null)

IMPORTANTE:
- El DataFrame `df` ya está filtrado: NO contiene podcasts (is_podcast == False).
- Los datos van desde {fecha_min} hasta {fecha_max}.
- Usa la fecha máxima del dataset como referencia para "hoy".
- Para "último trimestre", "últimos 3 meses", etc., calcula las fechas relativas a {fecha_max}.
- "Verano" = junio, julio, agosto. "Invierno" = diciembre, enero, febrero.

INSTRUCCIONES PARA EL CÓDIGO:
1. Genera código Python que use el DataFrame `df` y las librerías `px` (plotly.express) y `go` (plotly.graph_objects).
2. La última línea del código debe asignar la figura a la variable `fig`.
3. Siempre pon títulos descriptivos en español, etiquetas legibles y formatos claros.
4. Para rankings, usa barras horizontales con los valores ordenados.
5. Para evolución temporal, usa gráficos de línea.
6. Para distribuciones, usa gráficos de barras o donut.
7. Para patrones hora/día, usa heatmaps.
8. Formatea tiempos en horas o minutos, no en milisegundos.
9. Usa colores de la paleta de Plotly por defecto. No uses colores personalizados.

FORMATO DE RESPUESTA:
Responde SIEMPRE con un JSON válido y nada más. Sin markdown, sin explicaciones fuera del JSON.

Si la pregunta se puede responder con los datos:
{{"tipo": "grafico", "codigo": "código python aquí", "interpretacion": "texto breve en español explicando el resultado"}}

Si la pregunta NO se puede responder con los datos (predicciones, acciones, temas no relacionados):
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "explicación amable de por qué no puedo responder"}}

EJEMPLOS:

Pregunta: "¿Cuál es mi artista más escuchado?"
{{"tipo": "grafico", "codigo": "top = df.groupby('artist_name')['min_played'].sum().sort_values(ascending=False).head(10).reset_index()\\ntop.columns = ['Artista', 'Minutos']\\ntop['Horas'] = (top['Minutos'] / 60).round(1)\\nfig = px.bar(top, x='Horas', y='Artista', orientation='h', title='Top 10 artistas por tiempo de escucha (horas)', text='Horas')\\nfig.update_layout(yaxis={{'categoryorder': 'total ascending'}})", "interpretacion": "Tu artista más escuchado es X con Y horas de escucha."}}

Pregunta: "Recomiéndame música"
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "No puedo recomendarte música, pero puedo mostrarte qué artistas o canciones escuchas más para que descubras patrones en tus gustos."}}
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # Renombrar columnas largas para simplificar el código del LLM
    df = df.rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name",
    })

    # Columnas derivadas de tiempo
    df["min_played"] = (df["ms_played"] / 60000).round(2)
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.day_name()
    df["weekday_num"] = df["ts"].dt.weekday
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["is_weekend"] = df["weekday_num"] >= 5

    # Marcar podcasts
    df["is_podcast"] = df["track_name"].isna()

    # Filtrar podcasts para el análisis
    df = df[~df["is_podcast"]].reset_index(drop=True)

    return df


def build_prompt(df):
    """Inyecta información dinámica del dataset en el system prompt."""
    fecha_min = df["ts"].min().strftime("%Y-%m-%d")
    fecha_max = df["ts"].max().strftime("%Y-%m-%d")
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
def get_response(user_msg, system_prompt):
    """Envía la pregunta del usuario al LLM y devuelve la respuesta cruda."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def parse_response(raw):
    """
    Parsea la respuesta del LLM como JSON.
    Formato esperado:
    {
        "tipo": "grafico" | "fuera_de_alcance",
        "codigo": "código python (vacío si fuera_de_alcance)",
        "interpretacion": "texto breve explicando el resultado"
    }
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# FUNCIÓN DE EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
def execute_chart(code, df):
    """Ejecuta el código del LLM y devuelve la figura de Plotly."""
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

df = load_data()
system_prompt = build_prompt(df)

if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error(f"Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")
