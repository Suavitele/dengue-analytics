import streamlit as st
import pandas as pd
import joblib
import io

from pathlib import Path

# ============================================================
# Modelo de predicción de dengue | Aplicación académica
# ============================================================

st.set_page_config(
    page_title="Modelo de predicción de dengue",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO_LOGISTICO = BASE_DIR / "models" / "modelo_logistico.pkl"
RUTA_MODELO_NEURONAL = BASE_DIR / "models" / "modelo_red_neuronal.pkl"

COLUMNAS = [
    "AGE",
    "GENDER",
    "DURATION OF FEVER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY",
    "HB",
    "CREATININE",
    "PCV_ALTO",
    "PLAQUETAS_BAJAS",
    "WBC_ALTO",
    "ALT_ALTO"
]

# Resultados de la evaluación previamente realizada.
# Se conservan los valores de la versión anterior de app.py.
METRICAS = pd.DataFrame({
    "Métrica": [
        "Exactitud",
        "Precisión",
        "Sensibilidad",
        "Puntuación F1",
        "Área bajo la curva ROC"
    ],
    "Regresión logística": [
        0.852459,
        0.590909,
        1.000000,
        0.742857,
        0.969551
    ],
    "Red neuronal": [
        0.770492,
        0.476190,
        0.769231,
        0.588235,
        0.823718
    ]
})

# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>
    :root {
        --navy: #174B3A;
        --navy-2: #237A57;
        --teal: #269B68;
        --teal-soft: #EAF6EF;
        --ink: #263B32;
        --muted: #718078;
        --line: #DFEAE2;
        --surface: #FFFFFF;
        --background: #F5F9F6;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--background);
        color: var(--ink);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 250, 0.92);
    }

    [data-testid="stSidebar"] {
        display: none;
    }


    [data-testid="stSegmentedControl"] {
        background: transparent !important;
        border: 0 !important;
        gap: 8px !important;
    }

    [data-testid="stSegmentedControl"] button {
        background: #FFFFFF !important;
        color: #315544 !important;
        border: 1px solid #CFE0D4 !important;
        border-radius: 3px !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }

    [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: #237A57 !important;
        color: #FFFFFF !important;
        border-color: #237A57 !important;
    }

    [data-testid="stSegmentedControl"] button:hover {
        border-color: #269B68 !important;
        color: #174B3A !important;
    }

    [data-testid="stSegmentedControl"] button[aria-pressed="true"]:hover {
        color: #FFFFFF !important;
    }

    div[role="radiogroup"] {
        gap: 0.45rem;
        padding: 0.35rem 0 0.75rem 0;
    }

    div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 0.45rem 0.9rem;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: #EAF6EF;
        border-color: #269B68;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750 !important;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--ink);
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 15px 17px;
        box-shadow: 0 2px 8px rgba(24, 52, 71, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.83rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-weight: 750;
    }

    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1.25rem 1.35rem;
    }

    input, textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [role="combobox"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border-color: #D5DEE5 !important;
        border-radius: 3px !important;
        box-shadow: none !important;
    }

    /* Elimina fondos oscuros en flechas y controles internos. */
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] [data-testid],
    div[data-baseweb="select"] > div > div {
        color: #237A57 !important;
        fill: #237A57 !important;
    }

    div[data-baseweb="select"] [role="combobox"] {
        background: #FFFFFF !important;
    }

    [data-testid="stNumberInput"] button {
        background: #EAF6EF !important;
        color: #237A57 !important;
        border: 1px solid #D5E5D9 !important;
        border-radius: 3px !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: #D9EFE1 !important;
        color: #174B3A !important;
        border-color: #269B68 !important;
    }

    [data-testid="stNumberInput"] input {
        border-radius: 3px !important;
    }

    [data-testid="stCheckbox"] input {
        accent-color: #269B68 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"],
    div[role="option"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
    }

    [data-baseweb="popover"],
    ul[role="listbox"],
    div[role="option"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
    }

    [data-testid="stCheckbox"] label {
        color: var(--ink) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: var(--navy-2);
        color: #FFFFFF !important;
        border: 1px solid var(--navy-2);
        border-radius: 3px;
        min-height: 2.7rem;
        font-weight: 650;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: var(--teal);
        border-color: var(--teal);
        color: #FFFFFF !important;
    }

    .hero {
        background: linear-gradient(120deg, #174B3A 0%, #267653 100%);
        color: #FFFFFF;
        padding: 28px 30px;
        border-radius: 4px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(23, 52, 71, 0.12);
    }

    .hero-kicker {
        color: #BDE8CE;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: clamp(2rem, 4vw, 2.8rem);
    }

    .hero p {
        color: #DDEFE3;
        margin: 8px 0 0 0;
        max-width: 760px;
        line-height: 1.55;
    }

    .section-kicker {
        color: var(--teal);
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 18px 19px;
        height: 100%;
    }

    .info-card h3 {
        font-size: 1.06rem;
        margin: 0 0 7px 0;
    }

    .info-card p {
        color: #607181;
        font-size: 0.91rem;
        line-height: 1.55;
        margin: 0;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 20px;
        margin: 12px 0;
    }

    .result-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.72rem;
        font-weight: 750;
    }

    .result-value {
        color: var(--navy);
        font-size: 1.45rem;
        font-weight: 750;
        margin: 5px 0 8px 0;
    }

    .status-positive {
        color: #9A4B16;
        background: #FFF3E8;
        border: 1px solid #F5D7BD;
        padding: 8px 11px;
        border-radius: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .status-negative {
        color: #176B57;
        background: #EAF6F1;
        border: 1px solid #CDE9DC;
        padding: 8px 11px;
        border-radius: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.9rem;
    }



    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: var(--line);
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_resource
def cargar_modelos():
    """Carga una sola vez los dos modelos entrenados."""
    if not RUTA_MODELO_LOGISTICO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO_LOGISTICO}"
        )

    if not RUTA_MODELO_NEURONAL.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO_NEURONAL}"
        )

    modelo_logistico = joblib.load(RUTA_MODELO_LOGISTICO)
    modelo_neuronal = joblib.load(RUTA_MODELO_NEURONAL)

    return modelo_logistico, modelo_neuronal


def probabilidad_clase(modelo, entrada, clase_objetivo=1):
    """
    Obtiene la probabilidad asociada a la clase indicada,
    sin asumir que siempre ocupa la segunda posición.
    """
    probabilidades = modelo.predict_proba(entrada)[0]
    clases = list(modelo.classes_)

    if clase_objetivo not in clases:
        raise ValueError(
            f"El modelo no contiene la clase {clase_objetivo}."
        )

    indice = clases.index(clase_objetivo)
    return float(probabilidades[indice])


def crear_entrada(datos):
    """Construye el DataFrame respetando el orden del entrenamiento."""
    entrada = pd.DataFrame(
        [[datos[columna] for columna in COLUMNAS]],
        columns=COLUMNAS
    )
    return entrada


def mostrar_resultado(nombre, modelo, entrada):
    """Presenta la clasificación y sus probabilidades."""
    prediccion = int(modelo.predict(entrada)[0])
    prob_complicado = probabilidad_clase(modelo, entrada, 1)
    prob_no_complicado = probabilidad_clase(modelo, entrada, 0)

    st.markdown(f"#### {nombre}")

    if prediccion == 1:
        st.markdown(
            '<span class="status-positive">'
            'Clasificación estimada: dengue complicado'
            '</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-negative">'
            'Clasificación estimada: dengue no complicado'
            '</span>',
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)
    col1.metric(
        "Dengue no complicado",
        f"{prob_no_complicado * 100:.2f}%"
    )
    col2.metric(
        "Dengue complicado",
        f"{prob_complicado * 100:.2f}%"
    )

    st.caption("Probabilidad estimada por el modelo para cada clase.")
    st.progress(prob_complicado)

    return {
        "Modelo": nombre,
        "Clasificación": (
            "Dengue complicado"
            if prediccion == 1
            else "Dengue no complicado"
        ),
        "Probabilidad no complicado": prob_no_complicado,
        "Probabilidad complicado": prob_complicado
    }




# ============================================================
# PREDICCIÓN MASIVA DESDE EXCEL
# ============================================================

def preparar_excel_lote(archivo):
    """
    Acepta un Excel con las 18 variables ya preparadas o un Excel
    con las variables clínicas originales y los valores de laboratorio.
    Los indicadores derivados se calculan con los umbrales del formulario.
    """
    original = pd.read_excel(archivo)
    if original.empty:
        raise ValueError("El archivo Excel no contiene registros.")

    df = original.copy()
    # Normalizar encabezados: espacios exteriores y mayúsculas.
    df.columns = [str(c).strip().upper().replace("\\n", " ") for c in df.columns]

    alias = {
        "AGE": ["AGE", "EDAD"],
        "GENDER": ["GENDER", "GENERO", "GÉNERO", "SEXO"],
        "DURATION OF FEVER": [
            "DURATION OF FEVER", "FEVER DURATION", "DURATION OF FEVER (DAYS)",
            "DIAS DE FIEBRE", "DÍAS DE FIEBRE", "DURACION DE FIEBRE",
            "DURACIÓN DE FIEBRE"
        ],
        "HEADACHE": ["HEADACHE", "DOLOR DE CABEZA"],
        "MYALGIA": ["MYALGIA", "DOLOR MUSCULAR"],
        "ABDOMINAL PAIN": ["ABDOMINAL PAIN", "DOLOR ABDOMINAL"],
        "RASH": ["RASH", "ERUPCION", "ERUPCIÓN"],
        "VOMITING": ["VOMITING", "VOMITOS", "VÓMITOS"],
        "BREATHLESSNESS": ["BREATHLESSNESS", "DIFICULTAD RESPIRATORIA"],
        "BLEEDING": ["BLEEDING", "SANGRADO"],
        "BP AT ADMISSION": [
            "BP AT ADMISSION", "BLOOD PRESSURE AT ADMISSION",
            "PRESION ARTERIAL AL INGRESO", "PRESIÓN ARTERIAL AL INGRESO"
        ],
        "ORGANOMEGALY": ["ORGANOMEGALY", "ORGANOMEGALIA"],
        "HB": ["HB", "HEMOGLOBIN", "HEMOGLOBINA"],
        "CREATININE": ["CREATININE", "CREATININA"],
        "PCV": ["PCV", "HEMATOCRIT", "HEMATOCRITO"],
        "PLATELET COUNT": [
            "PLATELET COUNT", "PLATELETS", "PLAQUETAS",
            "RECUENTO DE PLAQUETAS"
        ],
        "HIGHEST WBC": [
            "HIGHEST WBC", "WBC", "WHITE BLOOD CELL",
            "LEUCOCITOS", "RECUENTO DE LEUCOCITOS"
        ],
        "ALT": ["ALT", "ALANINE AMINOTRANSFERASE"]
    }

    def buscar(nombre):
        for candidato in alias.get(nombre, [nombre]):
            if candidato in df.columns:
                return candidato
        return None

    X = pd.DataFrame(index=df.index)
    faltantes = []

    # Variables que deben venir ya codificadas igual que en entrenamiento.
    directas = [
        "AGE", "GENDER", "DURATION OF FEVER", "HEADACHE", "MYALGIA",
        "ABDOMINAL PAIN", "RASH", "VOMITING", "BREATHLESSNESS",
        "BLEEDING", "BP AT ADMISSION", "ORGANOMEGALY", "HB", "CREATININE"
    ]

    for col in directas:
        origen = buscar(col)
        if origen is None:
            faltantes.append(col)
        else:
            X[col] = pd.to_numeric(df[origen], errors="coerce")

    # Indicadores: se aceptan ya calculados o se derivan de los valores crudos.
    derivadas = {
        "PCV_ALTO": ("PCV", 42, "ge"),
        "PLAQUETAS_BAJAS": ("PLATELET COUNT", 75000, "le"),
        "WBC_ALTO": ("HIGHEST WBC", 7000, "ge"),
        "ALT_ALTO": ("ALT", 70, "ge")
    }

    for indicador, (variable, umbral, comparador) in derivadas.items():
        existente = buscar(indicador)
        if existente is not None:
            X[indicador] = pd.to_numeric(df[existente], errors="coerce")
            continue

        origen = buscar(variable)
        if origen is None:
            faltantes.append(f"{indicador} o {variable}")
            continue

        valores = pd.to_numeric(df[origen], errors="coerce")
        if variable == "ALT":
            valores = valores.fillna(26.0)
        if comparador == "ge":
            X[indicador] = (valores >= umbral).astype(float)
        else:
            X[indicador] = (valores <= umbral).astype(float)

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(faltantes)
            + ". El Excel debe incluir las 18 variables del modelo, "
              "o los valores originales PCV, PLATELET COUNT, HIGHEST WBC y ALT "
              "para calcular los indicadores."
        )

    # Misma imputación aplicada en la limpieza original.
    X["CREATININE"] = X["CREATININE"].fillna(0.5)

    # Convertir y validar todas las variables.
    for col in COLUMNAS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    columnas_invalidas = X.columns[X.isna().any()].tolist()
    if columnas_invalidas:
        raise ValueError(
            "Hay valores vacíos o no numéricos en: "
            + ", ".join(columnas_invalidas)
            + ". Completa esos datos en el Excel."
        )

    binarias = [
        "AGE", "GENDER", "HEADACHE", "MYALGIA", "ABDOMINAL PAIN",
        "RASH", "VOMITING", "BREATHLESSNESS", "BLEEDING",
        "BP AT ADMISSION", "ORGANOMEGALY", "PCV_ALTO",
        "PLAQUETAS_BAJAS", "WBC_ALTO", "ALT_ALTO"
    ]
    for col in binarias:
        if not X[col].isin([0, 1]).all():
            raise ValueError(
                f"La columna {col} debe contener únicamente 0 y 1, "
                "con la misma codificación usada para entrenar los modelos."
            )

    return original, X[COLUMNAS]


def predecir_excel_lote(modelo, X, nombre_modelo):
    pred = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    clases = list(modelo.classes_)
    if 0 not in clases or 1 not in clases:
        raise ValueError(f"{nombre_modelo}: no se encontraron las clases 0 y 1.")

    i0, i1 = clases.index(0), clases.index(1)
    return pd.DataFrame({
        f"{nombre_modelo} - Clasificación": [
            "Dengue complicado" if int(v) == 1 else "Dengue no complicado"
            for v in pred
        ],
        f"{nombre_modelo} - Prob. no complicado (%)":
            (probabilidades[:, i0] * 100).round(2),
        f"{nombre_modelo} - Prob. complicado (%)":
            (probabilidades[:, i1] * 100).round(2)
    })


# ============================================================
# CARGA DE MODELOS
# ============================================================

try:
    modelo_logistico, modelo_neuronal = cargar_modelos()
except Exception as error:
    st.error(
        "No fue posible cargar los modelos entrenados. "
        "Comprueba que existan los archivos dentro de la carpeta "
        "'models'."
    )
    st.code(
        "models/modelo_logistico.pkl\n"
        "models/modelo_red_neuronal.pkl"
    )
    st.exception(error)
    st.stop()


# ============================================================
# NAVEGACIÓN
# ============================================================

# Navegación horizontal en la parte superior
top_left, top_right = st.columns([3, 2])
with top_left:
    st.markdown("## Modelo de predicción de dengue")
    st.caption("DATA SCIENCE LAB · PROYECTO ACADÉMICO")

with top_right:
    if hasattr(st, "segmented_control"):
        pagina = st.segmented_control(
            "NAVEGACIÓN",
            options=["Inicio", "Predicción"],
            default="Inicio",
            label_visibility="collapsed",
            key="navegacion_principal"
        )
    else:
        pagina = st.radio(
            "NAVEGACIÓN",
            ["Inicio", "Predicción"],
            horizontal=True,
            label_visibility="collapsed"
        )

st.divider()


# ============================================================
# INICIO
# ============================================================

if pagina == "Inicio":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">Plataforma académica · Analítica de datos</div>
        <h1>Modelo de predicción de dengue</h1>
        <p>
            Herramienta académica de clasificación de dengue mediante
            regresión logística y redes neuronales artificiales.
            Explora los datos, ejecuta predicciones y compara el
            rendimiento de ambos modelos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Vista general</div>',
                unsafe_allow_html=True)
    st.subheader("El proyecto en cifras")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros procesados", "302")
    c2.metric("Variables predictoras", "18")
    c3.metric("Modelos entrenados", "2")
    c4.metric("Clases de salida", "2")

    st.write("")
    st.markdown('<div class="section-kicker">Modelos implementados</div>',
                unsafe_allow_html=True)
    st.subheader("Dos enfoques de clasificación")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>01 · Regresión logística</h3>
            <p>
                Algoritmo de clasificación supervisada que estima
                la probabilidad de pertenecer a una de dos categorías.
                En la evaluación registrada obtuvo una exactitud de
                85,25% y un AUC de 96,96%.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>02 · Red neuronal artificial</h3>
            <p>
                Modelo multicapa que aprende patrones a partir de
                las variables de entrada. En la evaluación registrada
                obtuvo una exactitud de 77,05% y un AUC de 82,37%.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-kicker">Propósito</div>',
                unsafe_allow_html=True)
    st.subheader("¿Qué permite hacer el sistema?")
    st.write(
        "El sistema recibe variables demográficas, síntomas y "
        "resultados de laboratorio. Con esa información construye "
        "un registro de entrada y lo procesa mediante uno o ambos "
        "modelos entrenados para observar y comparar sus estimaciones."
    )



# ============================================================
# PREDICCIÓN
# ============================================================

elif pagina == "Predicción":

    st.markdown('<div class="section-kicker">Evaluación individual</div>',
                unsafe_allow_html=True)
    st.title("Predicción")
    st.write(
        "Registra los datos demográficos, clínicos y de laboratorio. "
        "Después selecciona el modelo que deseas ejecutar."
    )

    with st.form("formulario_paciente", border=True):

        st.markdown("### 01 · Datos demográficos")
        col1, col2, col3 = st.columns(3)

        with col1:
            edad = st.selectbox(
                "Categoría de edad",
                ["Menor de 6 años", "Mayor de 6 años"]
            )

        with col2:
            genero = st.selectbox(
                "Sexo",
                ["Masculino", "Femenino"]
            )

        with col3:
            dias_fiebre = st.number_input(
                "Duración de la fiebre (días)",
                min_value=1,
                max_value=30,
                value=3,
                step=1
            )

        st.divider()
        st.markdown("### 02 · Síntomas y evaluación clínica")
        st.caption("Marca los síntomas o hallazgos presentes.")

        col1, col2, col3 = st.columns(3)

        with col1:
            headache = st.checkbox("Dolor de cabeza")
            myalgia = st.checkbox("Dolor muscular")
            abdominal_pain = st.checkbox("Dolor abdominal")

        with col2:
            rash = st.checkbox("Erupciones cutáneas")
            vomiting = st.checkbox("Vómitos")
            breathlessness = st.checkbox("Dificultad respiratoria")

        with col3:
            bleeding = st.checkbox("Sangrado")
            hypotension = st.checkbox("Hipotensión al ingreso")
            organomegaly = st.checkbox("Organomegalia")

        st.divider()
        st.markdown("### 03 · Resultados de laboratorio")
        st.caption(
            "Los cuatro indicadores derivados se calculan automáticamente "
            "con los mismos puntos de corte utilizados en limpiar.py."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            hb = st.number_input(
                "Hemoglobina (g/dL)",
                min_value=0.1,
                max_value=30.0,
                value=12.0,
                step=0.1
            )
            creatinina = st.number_input(
                "Creatinina (mg/dL)",
                min_value=0.01,
                max_value=20.0,
                value=0.50,
                step=0.01
            )

        with col2:
            pcv = st.number_input(
                "PCV (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
                help="Se marca PCV_ALTO cuando el valor es igual o superior a 42."
            )
            plaquetas = st.number_input(
                "Recuento de plaquetas",
                min_value=0,
                max_value=2000000,
                value=150000,
                step=1000,
                help="Se marca PLAQUETAS_BAJAS cuando el valor es igual o inferior a 75000."
            )

        with col3:
            wbc = st.number_input(
                "WBC máximo (células/cumm)",
                min_value=0,
                max_value=100000,
                value=6000,
                step=100,
                help="Se marca WBC_ALTO cuando el valor es igual o superior a 7000."
            )
            alt = st.number_input(
                "ALT (U/L)",
                min_value=0.0,
                max_value=10000.0,
                value=30.0,
                step=1.0,
                help="Se marca ALT_ALTO cuando el valor es igual o superior a 70."
            )

        st.divider()
        st.markdown("### 04 · Modelo de clasificación")

        seleccion = st.selectbox(
            "¿Qué modelo deseas ejecutar?",
            [
                "Comparar ambos modelos",
                "Regresión logística",
                "Red neuronal artificial"
            ]
        )

        ejecutar = st.form_submit_button(
            "Ejecutar análisis",
            type="primary",
            use_container_width=True
        )

    if ejecutar:

        # Codificación idéntica a la definida en limpiar.py:
        # AGE: 0 = menor de 6; 1 = mayor de 6.
        # GENDER: 0 = masculino; 1 = femenino.
        # Variables binarias: 1 = sí/presente; 0 = no/ausente.
        datos = {
            "AGE": 0 if edad == "Menor de 6 años" else 1,
            "GENDER": 0 if genero == "Masculino" else 1,
            "DURATION OF FEVER": int(dias_fiebre),
            "HEADACHE": int(headache),
            "MYALGIA": int(myalgia),
            "ABDOMINAL PAIN": int(abdominal_pain),
            "RASH": int(rash),
            "VOMITING": int(vomiting),
            "BREATHLESSNESS": int(breathlessness),
            "BLEEDING": int(bleeding),
            "BP AT ADMISSION": int(hypotension),
            "ORGANOMEGALY": int(organomegaly),
            "HB": float(hb),
            "CREATININE": float(creatinina),
            "PCV_ALTO": int(pcv >= 42),
            "PLAQUETAS_BAJAS": int(plaquetas <= 75000),
            "WBC_ALTO": int(wbc >= 7000),
            "ALT_ALTO": int(alt >= 70)
        }

        entrada = crear_entrada(datos)

        st.divider()
        st.markdown('<div class="section-kicker">Resultado del análisis</div>',
                    unsafe_allow_html=True)
        st.subheader("Estimaciones de los modelos")

        if seleccion == "Regresión logística":
            modelos_seleccionados = [
                ("Regresión logística", modelo_logistico)
            ]
        elif seleccion == "Red neuronal artificial":
            modelos_seleccionados = [
                ("Red neuronal artificial", modelo_neuronal)
            ]
        else:
            modelos_seleccionados = [
                ("Regresión logística", modelo_logistico),
                ("Red neuronal artificial", modelo_neuronal)
            ]

        resultados = []

        try:
            if len(modelos_seleccionados) == 2:
                c1, c2 = st.columns(2)
                for (nombre, modelo), columna in zip(
                    modelos_seleccionados, [c1, c2]
                ):
                    with columna:
                        resultados.append(
                            mostrar_resultado(nombre, modelo, entrada)
                        )
            else:
                nombre, modelo = modelos_seleccionados[0]
                resultados.append(
                    mostrar_resultado(nombre, modelo, entrada)
                )

            if len(resultados) == 2:
                st.divider()
                st.markdown("#### Comparación de estimaciones")
                comparacion = pd.DataFrame(resultados)
                comparacion["Probabilidad no complicado"] = (
                    comparacion["Probabilidad no complicado"] * 100
                ).map(lambda x: f"{x:.2f}%")
                comparacion["Probabilidad complicado"] = (
                    comparacion["Probabilidad complicado"] * 100
                ).map(lambda x: f"{x:.2f}%")
                st.dataframe(
                    comparacion,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            st.error("Ocurrió un problema al ejecutar la predicción.")
            st.exception(error)

    


# ============================================================
# PREDICCIÓN POR LOTE
# ============================================================

if pagina == "Predicción":
    st.divider()
    st.markdown('<div class="section-kicker">Procesamiento de varios registros</div>',
                unsafe_allow_html=True)
    st.subheader("Predicción por lote mediante Excel")
    st.write(
        "Carga un archivo .xlsx para obtener predicciones de varios pacientes "
        "con uno o ambos modelos."
    )

    archivo_lote = st.file_uploader(
        "Seleccionar archivo Excel",
        type=["xlsx"],
        key="archivo_excel_lote"
    )
    modelo_lote = st.selectbox(
        "Modelo para procesar el archivo",
        [
            "Comparar ambos modelos",
            "Regresión logística",
            "Red neuronal artificial"
        ],
        key="selector_modelo_lote"
    )

    if archivo_lote is not None:
        if st.button(
            "Procesar todos los registros",
            type="primary",
            use_container_width=True,
            key="boton_procesar_lote"
        ):
            try:
                with st.spinner("Validando datos y generando predicciones..."):
                    datos_originales, X_lote = preparar_excel_lote(archivo_lote)
                    resultado = datos_originales.copy()

                    if modelo_lote in [
                        "Comparar ambos modelos", "Regresión logística"
                    ]:
                        resultado = pd.concat(
                            [
                                resultado,
                                predecir_excel_lote(
                                    modelo_logistico, X_lote, "Regresión logística"
                                )
                            ],
                            axis=1
                        )

                    if modelo_lote in [
                        "Comparar ambos modelos", "Red neuronal artificial"
                    ]:
                        resultado = pd.concat(
                            [
                                resultado,
                                predecir_excel_lote(
                                    modelo_neuronal, X_lote,
                                    "Red neuronal artificial"
                                )
                            ],
                            axis=1
                        )

                st.success(
                    f"Se procesaron correctamente {len(resultado)} registros."
                )
                st.dataframe(
                    resultado,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("#### Resumen de resultados")
                c1, c2 = st.columns(2)

                if modelo_lote == "Regresión logística":
                    serie = resultado["Regresión logística - Clasificación"]
                elif modelo_lote == "Red neuronal artificial":
                    serie = resultado["Red neuronal artificial - Clasificación"]
                else:
                    serie = resultado["Regresión logística - Clasificación"]

                c1.metric(
                    "Dengue no complicado",
                    int((serie == "Dengue no complicado").sum())
                )
                c2.metric(
                    "Dengue complicado",
                    int((serie == "Dengue complicado").sum())
                )

                buffer_excel = io.BytesIO()
                with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                    resultado.to_excel(
                        writer, index=False, sheet_name="Predicciones"
                    )

                st.download_button(
                    "Descargar resultados en Excel",
                    data=buffer_excel.getvalue(),
                    file_name="resultados_prediccion_dengue.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

            except Exception as error:
                st.error(f"No se pudo procesar el archivo: {error}")

# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()
st.caption(
    "Modelo de predicción de dengue | Proyecto académico"
)
