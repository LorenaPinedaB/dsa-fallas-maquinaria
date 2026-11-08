"""
Tablero de diagnóstico de fallas en maquinaria rotativa (Entrega 2).

Qué hace:
  1. Carga la tabla de características (data/features_mafaulda.csv) y el
     manifiesto de partición de Daniel para quedarse con las grabaciones de PRUEBA
     (las que el modelo no vio al entrenar).
  2. Carga el modelo XGBoost V2 (models/xgboost_v2_bundle.joblib).
  3. El usuario elige una lectura, da clic en "Diagnosticar" y el tablero muestra:
     diagnóstico, probabilidad por clase, comparación con la condición normal
     y una recomendación de inspección.

En esta entrega el tablero carga el modelo localmente. En la entrega final solo
hay que cambiar el contenido de la función `predecir()` por una llamada a la API.

Cómo correrlo (desde la carpeta raíz del repositorio, con el ambiente activo):
    python dashboard/app.py
y abrir en el navegador: http://127.0.0.1:8050
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# ============================================================
# 1. RUTAS (relativas a la raíz del repositorio)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features_mafaulda.csv"
MANIFEST_PATH = ROOT / "reports" / "daniel" / "split_manifest_xgb_v1.csv"
MODEL_PATH = ROOT / "models" / "xgboost_v2_bundle.joblib"

# Información del modelo que se muestra en el tablero
MODEL_INFO = {
    "nombre": "XGBoost V2",
    "corrida_mlflow": "xgb_estratificado_v2_robusto",
    "f1_prueba": 0.988,
    "variables": 60,
}

# ============================================================
# 2. TEXTOS EN ESPAÑOL PARA CADA CLASE
# ============================================================
CLASES = {
    "normal": {
        "nombre": "Condición normal",
        "familia": "Sin falla",
        "recomendacion": "No se requiere intervención. Mantener el plan de monitoreo periódico.",
    },
    "imbalance": {
        "nombre": "Desbalanceo",
        "familia": "Desbalanceo",
        "recomendacion": "Revisar el balanceo del rotor: masas añadidas, acumulación de material o piezas sueltas.",
    },
    "horizontal_misalignment": {
        "nombre": "Desalineación horizontal",
        "familia": "Desalineación",
        "recomendacion": "Revisar la alineación horizontal del acople entre el motor y el eje.",
    },
    "vertical_misalignment": {
        "nombre": "Desalineación vertical",
        "familia": "Desalineación",
        "recomendacion": "Revisar la alineación vertical del acople (altura de soportes y calces).",
    },
    "underhang_outer_race": {
        "nombre": "Rod. interno · pista externa",
        "familia": "Falla de rodamiento interno (underhang)",
        "recomendacion": "Inspeccionar la pista externa del rodamiento interno (underhang).",
    },
    "underhang_cage_fault": {
        "nombre": "Rod. interno · jaula",
        "familia": "Falla de rodamiento interno (underhang)",
        "recomendacion": "Inspeccionar la jaula del rodamiento interno (underhang).",
    },
    "underhang_ball_fault": {
        "nombre": "Rod. interno · elementos rodantes",
        "familia": "Falla de rodamiento interno (underhang)",
        "recomendacion": "Inspeccionar los elementos rodantes del rodamiento interno (underhang).",
    },
    "overhang_outer_race": {
        "nombre": "Rod. externo · pista externa",
        "familia": "Falla de rodamiento externo (overhang)",
        "recomendacion": "Inspeccionar la pista externa del rodamiento externo (overhang).",
    },
    "overhang_cage_fault": {
        "nombre": "Rod. externo · jaula",
        "familia": "Falla de rodamiento externo (overhang)",
        "recomendacion": "Inspeccionar la jaula del rodamiento externo (overhang).",
    },
    "overhang_ball_fault": {
        "nombre": "Rod. externo · elementos rodantes",
        "familia": "Falla de rodamiento externo (overhang)",
        "recomendacion": "Inspeccionar los elementos rodantes del rodamiento externo (overhang).",
    },
}

INDICADORES = {
    "rms": "RMS",
    "peak_to_peak": "Pico a pico",
    "kurtosis": "Curtosis",
    "crest_factor": "Factor de cresta",
}
POSICIONES = {"underhang": "Rod. interno", "overhang": "Rod. externo"}
EJES = {"axial": "Axial", "radial": "Radial", "tangential": "Tangencial"}

# ============================================================
# 3. CARGA DE DATOS Y MODELO
# ============================================================
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
encoder = bundle["label_encoder"]
FEATURES = bundle["feature_columns"]


def construir_objetivo(df):
    """Misma definición que usó Daniel: fault_type + posición para rodamientos."""
    objetivo = df["fault_type"].astype(str).copy()
    es_rodamiento = df["bearing_position"].notna()
    objetivo[es_rodamiento] = (
        df.loc[es_rodamiento, "bearing_position"].astype(str) + "_" + objetivo[es_rodamiento]
    )
    return objetivo


def clave_archivo(ruta):
    """Parte de la ruta desde MAFAULDA, para que coincida en cualquier computador."""
    ruta = str(ruta).replace("\\", "/")
    return ruta.split("MAFAULDA/")[-1].lower()


datos = pd.read_csv(DATA_PATH)
datos["clase_real"] = construir_objetivo(datos)
datos["clave"] = datos["filepath"].map(clave_archivo)

if MANIFEST_PATH.exists():
    manifiesto = pd.read_csv(MANIFEST_PATH)
    manifiesto["clave"] = manifiesto["filepath"].map(clave_archivo)
    datos = datos.merge(manifiesto[["clave", "partition"]], on="clave", how="left")
    AVISO_PARTICION = "Lecturas del conjunto de prueba (no usadas para entrenar el modelo)."
else:
    datos["partition"] = "test"
    AVISO_PARTICION = "Aviso: no se encontró el manifiesto de partición; se muestran todas las lecturas."

prueba = datos[datos["partition"] == "test"].sort_values("nominal_speed_rpm").reset_index(drop=True)
prueba["id_lectura"] = [f"Lectura {i + 1:03d}" for i in range(len(prueba))]

# Referencia de condición normal: solo grabaciones normales de ENTRENAMIENTO
normales_ref = datos[(datos["clase_real"] == "normal") & (datos["partition"] == "train")]
if normales_ref.empty:
    normales_ref = datos[datos["clase_real"] == "normal"]


# ============================================================
# 4. PREDICCIÓN (en la entrega final, esta función llamará a la API)
# ============================================================
def predecir(caracteristicas):
    """Recibe un DataFrame de una fila con las 60 variables y devuelve
    (clase_predicha, diccionario {clase: probabilidad})."""
    probabilidades = pipeline.predict_proba(caracteristicas[FEATURES])[0]
    clases = encoder.inverse_transform(np.arange(len(probabilidades)))
    prob = dict(zip(clases, probabilidades))
    clase = max(prob, key=prob.get)
    return clase, prob


def referencia_normal(velocidad_rpm, n=5):
    """Mediana de las n grabaciones normales con velocidad más cercana."""
    cercanas = normales_ref.iloc[
        (normales_ref["nominal_speed_rpm"] - velocidad_rpm).abs().argsort()[:n]
    ]
    return cercanas


# ============================================================
# 5. ESTILOS (colores del mockup)
# ============================================================
COLOR_ACENTO = "#1F5F8B"
COLOR_TEXTO = "#1B1F23"
COLOR_SECUNDARIO = "#5A6169"
COLOR_BORDE = "#DADDD8"
COLOR_FONDO = "#F3F4F1"
COLOR_ALERTA = "#A84A15"
FUENTE = "'IBM Plex Sans', system-ui, sans-serif"
FUENTE_MONO = "'IBM Plex Mono', monospace"

TARJETA = {
    "background": "white",
    "border": f"1px solid {COLOR_BORDE}",
    "borderRadius": "8px",
    "padding": "20px",
}
ETIQUETA = {
    "fontSize": "12px",
    "fontWeight": 600,
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "color": COLOR_SECUNDARIO,
    "marginBottom": "10px",
}



def num(x, dec=2):
    """Formato colombiano: punto para miles y coma para decimales."""
    texto = f"{x:,.{dec}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def chip(texto, fondo, color):
    return html.Span(
        texto,
        style={"fontSize": "12px", "padding": "6px 10px", "borderRadius": "4px",
               "background": fondo, "color": color, "whiteSpace": "nowrap"},
    )


# ============================================================
# 6. DISEÑO DEL TABLERO
# ============================================================
app = Dash(
    __name__,
    title="Diagnóstico de fallas",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700"
        "&family=IBM+Plex+Mono:wght@400;500&display=swap"
    ],
)

opciones_lectura = [
    {"label": f"{fila.id_lectura} · {num(fila.nominal_speed_rpm, 0)} rpm", "value": fila.clave}
    for fila in prueba.itertuples()
]

encabezado = html.Header(
    style={"background": COLOR_TEXTO, "color": "white", "padding": "18px 32px",
           "display": "flex", "flexWrap": "wrap", "gap": "12px",
           "justifyContent": "space-between", "alignItems": "center"},
    children=[
        html.Div([
            html.Div("Diagnóstico de fallas · Maquinaria rotativa",
                     style={"fontSize": "20px", "fontWeight": 600}),
            html.Div("Clasificación del tipo de falla a partir de una lectura de vibración de 5 s",
                     style={"fontSize": "13px", "color": "#C9CDD2"}),
        ]),
        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
            chip(f"Modelo: {MODEL_INFO['nombre']}", "#2A3036", "#E3E6E9"),
            chip(f"F1 macro en prueba: {num(MODEL_INFO['f1_prueba'], 3)}", "#2B3A2F", "#BFE3C6"),
            chip("Modelo cargado localmente", "#4A3A12", "#F2D99A"),
        ]),
    ],
)

panel_lectura = html.Aside(
    style={"flex": "1 1 280px", "maxWidth": "340px", "display": "flex",
           "flexDirection": "column", "gap": "16px"},
    children=[
        html.Section(style=TARJETA, children=[
            html.Div("1 · Lectura", style=ETIQUETA),
            html.Label("Lectura de vibración", htmlFor="lectura",
                       style={"fontSize": "13px", "color": "#3A4047"}),
            dcc.Dropdown(id="lectura", options=opciones_lectura,
                         value=opciones_lectura[len(opciones_lectura) // 2]["value"],
                         clearable=False, style={"marginTop": "6px", "fontSize": "13px"}),
            html.Div(AVISO_PARTICION,
                     style={"fontSize": "12px", "color": COLOR_SECUNDARIO, "marginTop": "8px"}),
            html.Div(id="velocidad", style={"marginTop": "14px"}),
            html.Button("Diagnosticar", id="boton", n_clicks=0, style={
                "marginTop": "16px", "width": "100%", "minHeight": "44px", "border": "none",
                "borderRadius": "6px", "background": COLOR_ACENTO, "color": "white",
                "fontSize": "15px", "fontWeight": 600, "cursor": "pointer", "fontFamily": FUENTE,
            }),
        ]),
        html.Section(style=TARJETA, children=[
            html.Div("Cómo funciona", style=ETIQUETA),
            html.P(
                "El tablero toma las 60 características estadísticas de la lectura "
                "(6 canales de vibración × 10 indicadores) y las pasa al modelo, que "
                "devuelve la probabilidad de cada una de las 10 condiciones.",
                style={"fontSize": "13px", "lineHeight": 1.5, "color": "#3A4047", "margin": 0},
            ),
            html.P(
                f"Corrida en MLflow: {MODEL_INFO['corrida_mlflow']}",
                style={"fontSize": "12px", "fontFamily": FUENTE_MONO, "color": COLOR_TEXTO,
                       "background": COLOR_FONDO, "padding": "8px 10px", "borderRadius": "4px",
                       "marginBottom": 0},
            ),
        ]),
    ],
)

contenido = html.Main(
    style={"flex": "999 1 560px", "minWidth": 0, "display": "flex",
           "flexDirection": "column", "gap": "20px"},
    children=[
        html.Section(id="diagnostico", style=TARJETA),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "20px"}, children=[
            html.Section(style={**TARJETA, "flex": "1 1 380px", "minWidth": 0}, children=[
                html.Div("3 · Probabilidad por clase", style=ETIQUETA),
                dcc.Graph(id="grafico", config={"displayModeBar": False},
                          style={"height": "340px"}),
            ]),
            html.Section(style={**TARJETA, "flex": "1 1 380px", "minWidth": 0}, children=[
                html.Div("4 · Lectura frente a condición normal", style=ETIQUETA),
                dcc.RadioItems(
                    id="eje", value="radial", inline=True,
                    options=[{"label": f" {v}", "value": k} for k, v in EJES.items()],
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px", "fontSize": "13px"},
                ),
                html.Div(id="tabla", style={"marginTop": "10px", "overflowX": "auto"}),
                html.Div(id="nota_referencia",
                         style={"fontSize": "12px", "color": COLOR_SECUNDARIO, "marginTop": "8px"}),
            ]),
        ]),
        html.Section(id="recomendacion", style=TARJETA),
    ],
)

app.layout = html.Div(
    style={"fontFamily": FUENTE, "background": COLOR_FONDO, "minHeight": "100vh",
           "color": COLOR_TEXTO},
    children=[
        encabezado,
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "24px",
                        "padding": "24px 32px"},
                 children=[panel_lectura, contenido]),
    ],
)


# ============================================================
# 7. FUNCIONES QUE ARMAN CADA BLOQUE
# ============================================================
def bloque_diagnostico(fila, clase, prob):
    p = prob[clase]
    if p >= 0.8:
        confianza = chip("Confianza alta", "#E3EEF5", "#163F5C")
    elif p >= 0.5:
        confianza = chip("Confianza media", "#FBF0D9", "#6B4A00")
    else:
        confianza = chip("Confianza baja", "#F8E4DA", "#7A2E0B")

    acierto = fila["clase_real"] == clase
    verificacion = html.Div(
        [
            html.Span("Falla real (conjunto de prueba): ", style={"color": COLOR_SECUNDARIO}),
            html.Span(CLASES[fila["clase_real"]]["nombre"], style={"fontWeight": 600}),
            html.Span("  ✓ El modelo acertó" if acierto else "  ✗ El modelo se equivocó",
                      style={"color": "#2E6B4F" if acierto else COLOR_ALERTA,
                             "fontWeight": 600, "marginLeft": "8px"}),
        ],
        style={"fontSize": "13px", "marginTop": "10px"},
    )

    return [
        html.Div("2 · Diagnóstico", style=ETIQUETA),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "20px",
                        "justifyContent": "space-between", "alignItems": "center"}, children=[
            html.Div([
                html.Div(CLASES[clase]["nombre"],
                         style={"fontSize": "30px", "fontWeight": 700}),
                html.Div(f"Familia: {CLASES[clase]['familia']}",
                         style={"fontSize": "14px", "color": "#3A4047"}),
                verificacion,
            ]),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                html.Div([
                    html.Div(f"{num(p * 100, 1)}%",
                             style={"fontFamily": FUENTE_MONO, "fontSize": "38px",
                                    "color": COLOR_ACENTO, "textAlign": "right"}),
                    html.Div("probabilidad", style={"fontSize": "13px",
                                                    "color": COLOR_SECUNDARIO,
                                                    "textAlign": "right"}),
                ]),
                confianza,
            ]),
        ]),
    ]


def grafico_probabilidades(clase, prob):
    orden = sorted(prob, key=prob.get)  # de menor a mayor (la mayor queda arriba)
    valores = [prob[c] * 100 for c in orden]
    colores = [COLOR_ACENTO if c == clase else "#9AA3AB" for c in orden]
    fig = go.Figure(go.Bar(
        x=valores, y=[CLASES[c]["nombre"] for c in orden], orientation="h",
        marker_color=colores,
        text=[f"{num(v, 1)}%" for v in valores], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=30), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=COLOR_TEXTO),
        xaxis=dict(range=[0, 118], title="Probabilidad (%)", gridcolor="#ECEDEA",
                   zeroline=False),
        yaxis=dict(automargin=True),
        showlegend=False,
    )
    return fig


def tabla_comparacion(fila, eje):
    ref = referencia_normal(fila["nominal_speed_rpm"])
    estilo_celda = {"padding": "8px 6px", "borderBottom": "1px solid #ECEDEA",
                    "textAlign": "right", "fontFamily": FUENTE_MONO}
    estilo_titulo = {"padding": "8px 6px", "borderBottom": f"1px solid {COLOR_BORDE}",
                     "fontWeight": 500, "color": COLOR_SECUNDARIO, "textAlign": "right"}
    filas = []
    for pos, nombre_pos in POSICIONES.items():
        for ind, nombre_ind in INDICADORES.items():
            col = f"{pos}_{eje}_{ind}"
            valor = fila[col]
            referencia = ref[col].median()
            if ind == "kurtosis":
                # La curtosis normal es cercana a 0: se muestra la diferencia, no el %
                diferencia = valor - referencia
                texto_variacion = ("+" if diferencia > 0 else "") + num(diferencia) + " (dif.)"
                destacado = abs(diferencia) >= 1
            else:
                variacion = (valor - referencia) / abs(referencia) * 100
                texto_variacion = ("+" if variacion > 0 else "") + f"{num(variacion, 0)}%"
                destacado = abs(variacion) >= 50
            filas.append(html.Tr([
                html.Td(f"{nombre_pos} · {nombre_ind}",
                        style={**estilo_celda, "textAlign": "left", "fontFamily": FUENTE}),
                html.Td(num(valor), style=estilo_celda),
                html.Td(num(referencia), style=estilo_celda),
                html.Td(texto_variacion, style={
                    **estilo_celda, "color": COLOR_ALERTA if destacado else "#3A4047",
                    "fontWeight": 600 if destacado else 400}),
            ]))
    tabla = html.Table(
        [html.Thead(html.Tr([
            html.Th("Indicador", style={**estilo_titulo, "textAlign": "left"}),
            html.Th("Lectura", style=estilo_titulo),
            html.Th("Ref. normal", style=estilo_titulo),
            html.Th("Variación", style=estilo_titulo),
        ])), html.Tbody(filas)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
    )
    nota = (f"Referencia: mediana de las {len(ref)} grabaciones normales de entrenamiento "
            f"con velocidad más cercana ({num(ref['nominal_speed_rpm'].min(), 0)}–"
            f"{num(ref['nominal_speed_rpm'].max(), 0)} rpm). En rojo, variaciones de ±50% o más "
            f"(en curtosis, diferencias de ±1 o más, porque su valor normal es cercano a 0).")
    return tabla, nota


def bloque_recomendacion(clase):
    return [
        html.Div("5 · Recomendación", style=ETIQUETA),
        html.Div(CLASES[clase]["recomendacion"],
                 style={"fontSize": "17px", "fontWeight": 600}),
        html.Div("Orientación para el técnico; el diagnóstico debe confirmarse con inspección en sitio.",
                 style={"fontSize": "13px", "color": "#3A4047", "marginTop": "4px"}),
    ]


# ============================================================
# 8. INTERACTIVIDAD
# ============================================================
@app.callback(
    Output("velocidad", "children"),
    Input("lectura", "value"),
)
def mostrar_velocidad(clave):
    fila = prueba.loc[prueba["clave"] == clave].iloc[0]
    return html.Div(
        [html.Span("Velocidad de giro"),
         html.Span(f"{num(fila['nominal_frequency_hz'], 1)} Hz · {num(fila['nominal_speed_rpm'], 0)} rpm",
                   style={"fontFamily": FUENTE_MONO})],
        style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "justifyContent": "space-between", "fontSize": "13px",
               "padding": "10px 12px", "background": COLOR_FONDO, "borderRadius": "6px"},
    )


@app.callback(
    Output("diagnostico", "children"),
    Output("grafico", "figure"),
    Output("tabla", "children"),
    Output("nota_referencia", "children"),
    Output("recomendacion", "children"),
    Input("boton", "n_clicks"),
    Input("eje", "value"),
    State("lectura", "value"),
)
def diagnosticar(_clics, eje, clave):
    fila = prueba.loc[prueba["clave"] == clave].iloc[0]
    clase, prob = predecir(prueba.loc[prueba["clave"] == clave])
    tabla, nota = tabla_comparacion(fila, eje)
    return (
        bloque_diagnostico(fila, clase, prob),
        grafico_probabilidades(clase, prob),
        tabla,
        nota,
        bloque_recomendacion(clase),
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
