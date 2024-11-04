# Importar bibliotecas necesarias
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import psycopg2

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
    WHERE {job} = 1
    AND age BETWEEN {age[0]} AND {age[1]}
    AND {education} = 1
    """
    query2 = f"""
    SELECT y, count(y) as count
    FROM bank_clean
    WHERE {job} = 1
    AND age BETWEEN {age[0]} AND {age[1]}
    AND {education} = 1
    GROUP BY y
    """
    if education == 'education_unknown':
        query = query.replace(f"AND {education} = 1", "")
        query2 = query2.replace(f"AND {education} = 1", "")
    
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

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
