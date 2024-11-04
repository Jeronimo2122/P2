# Importar bibliotecas necesarias
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Configuración de la conexión a la base de datos
DATABASE_CONFIG = {
    'dbname': "proy2",
    'user': "postgres",
    'password': "Lab9sqljuan",
    'host': "database-juan.c139i8kqq4e8.us-east-1.rds.amazonaws.com",
    'port': '5432'
}

# Establecer conexión a la base de datos
engine = psycopg2.connect(**DATABASE_CONFIG)

# Función para cargar y ajustar el scaler con los datos de entrenamiento
def cargar_scaler(X_train):
    scaler = StandardScaler()
   
    scaler.fit_transform(X_train)
    return scaler

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model2.h5')
X_train = np.load("X_train.npy")  # Cargar los datos de entrenamiento

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1("Estudio de aceptación de productos bancarios", style={'textAlign': 'center'}),
    
    # Contenedor principal con estilo flex
    html.Div(style={'display': 'flex'}, children=[
        
        # Panel izquierdo para selección de trabajo y rango de edad
        html.Div(style={
            'flex': '1',
            'padding': '10px',
            'display': 'flex',
            'flexDirection': 'column',
            'border': '3px solid #000000',
            'borderRadius': '10px',
            'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',
            'justify-conten': 'center'
        }, children=[
            html.H2("Seleccione el trabajo que desea estudiar"),
            html.P("Tenga en cuenta que los trabajos están divididos en 3 categorías:"),
            html.P("• Blue Collar: Trabajos manuales o de fábrica"),
            html.P("• Pink Collar: Trabajos de servicios"),
            html.P("• White Collar: Trabajos de oficina o gerenciales"),
            
            dcc.Dropdown(
                id='job-dropdown',
                options=[
                    {'label': 'Blue Collar', 'value': 'job_blue_collar'},
                    {'label': 'Pink Collar', 'value': 'job_pink_collar'},
                    {'label': 'White Collar', 'value': 'job_white_collar'},
                    {'label': 'Desconocido', 'value': 'job_unknown'},
                ],
                value='job_blue_collar',
                clearable=False
            ),
            
            html.Br(),
            html.H2("Seleccione el rango de edad"),
            dcc.RangeSlider(
                id='age-slider',
                min=0,
                max=100,
                step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                value=[40, 50]
            ),

            html.Br(),
            html.H2("Seleccione el nivel de educación"),
            dcc.Dropdown(
                id='education-dropdown',
                options=[
                    {'label': 'Primaria', 'value': 'education_primary'},
                    {'label': 'Secundaria', 'value': 'education_secondary'},
                    {'label': 'Terciaria', 'value': 'education_tertiary'},
                    {'label': 'Desconocida', 'value': 'education_unknown'},
                ],
                value='education_primary',
                clearable=False
            ),
        ]),
        
        # Panel derecho para mostrar resultados
        html.Div(style={
            'flex': '2',
            'padding': '10px',
            'display': 'flex',
            'flexDirection': 'column',
            'borderRadius': '10px',
            'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'
        }, children=[
            # Tasa de conversion de clientes
            dcc.Graph(id='conversion-rate'),
        ]),
    ]),
    html.Br(),
    # Promedio de balance de la cuenta
    html.H2("Promedio de balance de la cuenta", id='balance-avg'),
    html.Br(),
    html.Div(style={'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
        # Histograma de balance de la cuenta
        dcc.Graph(id='balance-histogram'),
    ]),

    html.Br(),
    html.H1("Predicción de conversion de los clientes", style={'textAlign': 'center'}),
    html.Div(children=[
        html.Div(
            style={
                'flex': '1',
                'padding': '10px',
                'display': 'flex',
                'flexDirection': 'row',  # Cambiado a 'row' para columnas
                'flexWrap': 'wrap',      # Permite que los elementos se ajusten automáticamente
                'border': '3px solid #000000',
                'borderRadius': '10px',
                'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',
                'justifyContent': 'space-around'
            },
            children=[
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Datos del cliente"),
                    html.P("Ingrese los datos del cliente para predecir si aceptará el producto bancario:")
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("• Edad: Edad del cliente"),
                    dcc.Input(
                        id='edad-input',
                        type='number',
                        placeholder='Edad del cliente',
                        min=0,
                        max=100,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=40
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("• Default: ¿Tiene crédito en default? (0 = no, 1 = sí)"),
                    dcc.Input(
                        id='default-input',
                        type='number',
                        placeholder='Crédito en default',
                        min=0,
                        max=1,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=0
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("• Balance: Saldo promedio anual"),
                    dcc.Input(
                        id='balance-input',
                        type='number',
                        placeholder='Saldo promedio anual',
                        min=0,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=1500
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Housing: ¿Tiene préstamo de vivienda? (0 = no, 1 = sí)"),
                    dcc.Input(
                        id='housing-input',
                        type='number',
                        placeholder='Préstamo de vivienda',
                        min=0,
                        max=1,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=1
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Loan: ¿Tiene préstamo personal? (0 = no, 1 = sí)"),
                    dcc.Input(
                        id='loan-input',
                        type='number',
                        placeholder='Préstamo personal',
                        min=0,
                        max=1,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=0
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Day: Día del último contacto"),
                    dcc.Input(
                        id='day-input',
                        type='number',
                        placeholder='Día del último contacto',
                        min=1,
                        max=31,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=15
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Month: Mes del último contacto"),
                    dcc.Input(
                        id='month-input',
                        type='number',
                        placeholder='Mes del último contacto',
                        min=1,
                        max=12,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=5
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Duration: Duración del último contacto en segundos"),
                    dcc.Input(
                        id='duration-input',
                        type='number',
                        placeholder='Duración del último contacto',
                        min=0,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=300
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Campaign: Número de contactos en esta campaña"),
                    dcc.Input(
                        id='campaign-input',
                        type='number',
                        placeholder='Número de contactos en esta campaña',
                        min=1,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=2
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H4("Previous: Número de contactos antes de esta campaña"),
                    dcc.Input(
                        id='previous-input',
                        type='number',
                        placeholder='Número de contactos antes de esta campaña',
                        min=0,
                        step=1,
                        style={'marginBottom': '10px'},
                        value=0
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Seleccione el trabajo del cliente"),
                    dcc.Dropdown(
                        id='job-dropdown2',
                        options=[
                            {'label': 'Blue Collar', 'value': 'job_blue_collar'},
                            {'label': 'Pink Collar', 'value': 'job_pink_collar'},
                            {'label': 'White Collar', 'value': 'job_white_collar'},
                            {'label': 'Desconocido', 'value': 'job_unknown'},
                        ],
                        value='job_blue_collar',
                        clearable=False
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Seleccione el estado civil del cliente"),
                    dcc.Dropdown(
                        id='marital-dropdown',
                        options=[
                            {'label': 'Casado', 'value': 'marital_married'},
                            {'label': 'Soltero', 'value': 'marital_single'},
                            {'label': 'Divorsiado/viudo', 'value': 'marital_unknown'},
                        ],
                        value='marital_married',
                        clearable=False
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Seleccione el nivel de educación del cliente"),
                    dcc.Dropdown(
                        id='education-dropdown2',
                        options=[
                            {'label': 'Primaria', 'value': 'education_primary'},
                            {'label': 'Secundaria', 'value': 'education_secondary'},
                            {'label': 'Terciaria', 'value': 'education_tertiary'},
                            {'label': 'Desconocida', 'value': 'education_unknown'},
                        ],
                        value='education_primary',
                        clearable=False
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Seleccione el método de contacto"),
                    dcc.Dropdown(
                        id='contact-dropdown',
                        options=[
                            {'label': 'Celular', 'value': 'contact_cellular'},
                            {'label': 'Telefono', 'value': 'contact_telephone'},
                            {'label': 'Desconocido', 'value': 'contact_unknown'},
                        ],
                        value='contact_cellular',
                        clearable=False
                    ),
                ]),
                html.Div(style={'flex': '1 1 45%', 'margin': '10px'}, children=[
                    html.H2("Seleccione el resultado de la campaña anterior"),
                    dcc.Dropdown(
                        id='poutcome-dropdown',
                        options=[
                            {'label': 'Fracaso', 'value': 'poutcome_failure'},
                            {'label': 'Otro', 'value': 'poutcome_other'},
                            {'label': 'Éxito', 'value': 'poutcome_success'},
                            {'label': 'Desconocido', 'value': 'poutcome_unknown'},
                        ],
                        value='poutcome_other',
                        clearable=False
                    ),
                ]),
            ]
        ),
        html.Div(style={'flex': '1', 'padding': '10px'}, children=[
            html.H2("Resultado de la predicción", style={'textAlign': 'center'}),
            html.H2(id='prediccion-output'),
        ]),
    ]),
])

# Callback para actualizar el promedio de balance de la cuenta
@app.callback(
    Output('balance-histogram', 'figure'),
    Output('balance-avg', 'children'),
    Output('conversion-rate', 'figure'),
    Input('job-dropdown', 'value'),
    Input('age-slider', 'value'),
    Input('education-dropdown', 'value')
)
def update_balance(job, age, education):
    # Consulta SQL para obtener los balances
    query = f"""
    SELECT balance
    FROM bank_clean
    WHERE age BETWEEN {age[0]} AND {age[1]}
    AND {job} = 1
    AND {education} = 1
    """
    query2 = f"""
    SELECT y, count(y) as count
    FROM bank_clean
    WHERE age BETWEEN {age[0]} AND {age[1]}
    AND {job} = 1
    AND {education} = 1
    GROUP BY y
    """
    if education == 'education_unknown':
        query = query.replace(f"AND {education} = 1", "")
        query2 = query2.replace(f"AND {education} = 1", "")
    
    if job == 'job_unknown':
        query = query.replace(f"AND {job} = 1", "")
        query2 = query2.replace(f"AND {job} = 1", "")
    
    # Leer datos de la base de datos
    df = pd.read_sql(query, engine)
    df2 = pd.read_sql(query2, engine)
    
    # Calcular el promedio del balance
    avg = df['balance'].mean() if not df.empty else 0
    
    # Crear el histograma
    fig = px.histogram(
        df,
        x='balance',
        nbins=20,
        title='Frecuencia de Balance de la Cuenta',
        labels={'balance': 'Balance de la Cuenta', 'count': 'Número de Personas'}
    )

    # Mejoras estéticas
    fig.update_traces(marker=dict(color='#66c2a5', line=dict(color='#FFFFFF', width=1)))  # Colores y bordes del histograma
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis=dict(
            title_standoff=10, 
            showgrid=False,
        ),
        yaxis=dict(
            title_standoff=10,
            showgrid=False, 
        ),
        margin=dict(t=50, b=0, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Crear la tasa de conversion
    fig2 = px.pie(
        df2, 
        values='count', 
        names='y', 
        title='Tasa de Conversión de Clientes',
        #labels={'count': 'Número de Personas'}
    )

    # Mejoras estéticas
    fig2.update_traces(
        textposition='inside',
        hoverinfo='label+percent', 
        marker=dict(
            colors=['#66c2a5', '#fc8d62'],
            line=dict(color='#FFFFFF', width=1)
        )
    )

    # Actualizar el diseño
    fig2.update_layout(
        title_font_size=20,
        title_x=0.5,
        margin=dict(t=50, b=0, l=0, r=0),
    )
    
    # Devolver figura y mensaje de promedio
    return fig, f"El promedio del balance para una persona entre {age[0]} y {age[1]} años con un trabajo de tipo {job} es de {avg:.2f}", fig2

# Callback para predecir la aceptación de un cliente
@app.callback(
    Output('prediccion-output', 'children'),
    Input('edad-input', 'value'),
    Input('default-input', 'value'),
    Input('balance-input', 'value'),
    Input('housing-input', 'value'),
    Input('loan-input', 'value'),
    Input('day-input', 'value'),
    Input('month-input', 'value'),
    Input('duration-input', 'value'),
    Input('campaign-input', 'value'),
    Input('previous-input', 'value'),
    Input('job-dropdown2', 'value'),
    Input('marital-dropdown', 'value'),
    Input('education-dropdown2', 'value'),
    Input('contact-dropdown', 'value'),
    Input('poutcome-dropdown', 'value')
)
def predecir_aceptacion(edad, default, balance, housing, loan, day, month, duration, campaign, previous,
                        job, marital, education, contact, poutcome):
    # Datos del cliente
    nuevo_cliente = [
        edad,  # age: Edad del cliente
        default,   # default: ¿Tiene crédito en default?
        balance, # balance: Saldo promedio anual en euros
        housing,   # housing: ¿Tiene préstamo de vivienda?
        loan,   # loan: ¿Tiene préstamo personal?
        day,  # day: Día del último contacto
        month,   # month: Mes del último contacto
        duration, # duration: Duración del último contacto en segundos
        campaign,   # campaign: Número de contactos en esta campaña
        previous,   # previous: Número de contactos antes de esta campaña
        
        # job dummies
        1 if job == 'job_blue_collar' else 0,   # job_blue-collar
        1 if job == 'job_pink_collar' else 0,   # job_pink-collar
        1 if job == 'job_white_collar' else 0,   # job_white-collar
        
        # marital dummies
        1 if marital == 'marital_married' else 0,   # marital_married
        1 if marital == 'marital_single' else 0,   # marital_single
        
        # education dummies
        1 if education == 'education_primary' else 0,   # education_primary
        1 if education == 'education_secondary' else 0,   # education_secondary
        1 if education == 'education_tertiary' else 0,   # education_tertiary
        
        # contact dummies
        1 if contact == 'contact_cellular' else 0,   # contact_cellular
        1 if contact == 'contact_unknown' else 0,   # contact_telephone
        
        # poutcome dummies
        1 if poutcome == 'poutcome_failure' else 0,   # poutcome_failure
        1 if poutcome == 'poutcome_other' else 0,   # poutcome_other
        1 if poutcome == 'poutcome_success' else 0,   # poutcome_success
    ]

    # Convertir los datos del cliente nuevo a un array de numpy
    nuevo_cliente = np.array([nuevo_cliente])

    # Escalar los datos usando el scaler ajustado en los datos de entrenamiento
    scaler = cargar_scaler(X_train)
    nuevo_cliente_scaled = scaler.transform(nuevo_cliente)

    # Realizar la predicción
    prediccion = model.predict(nuevo_cliente_scaled)

    # Clasificar la predicción (0 o 1)
    resultado = (prediccion > 0.5).astype("int32")

    color = '#66c2a5' if resultado[0][0] == 1 else '#fc8d62'

    # Devolver el resultado
    respuesta = html.Div(style={'textAlign': 'center', 'color': color}, children=[
        html.H3(f"El cliente aceptara el producto bancario con una probabilidad del {prediccion[0][0]:.2f}"),
        html.P(f"Con la informacion proporcionada, el modelo predice que el cliente {'aceptará' if resultado[0][0] == 1 else 'no aceptará'} el producto bancario.")
    ])
    return respuesta

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
