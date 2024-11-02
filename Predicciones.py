import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Función para crear el scaler a partir de los datos de entrenamiento
def scaler():
    # Crear un scaler para normalizar los datos
    scaler = StandardScaler()
    # Cargar los datos de entrenamiento para ajustar el scaler
    X_train = np.load("X_train.npy")
    scaler.fit(X_train)
    return scaler

# Importar modelo entrenado
model_imp = tf.keras.models.load_model('model2.h5')

def get_user_input():
    # Solicitar cada variable al usuario
    age = int(input("Edad del cliente (age): "))
    job = input("Tipo de trabajo (job): admin., unknown, unemployed, management, housemaid, entrepreneur, student, blue-collar, self-employed, retired, technician, services: ")
    marital = input("Estado civil (marital): married, divorced, single: ")
    education = input("Nivel de educación (education): unknown, secondary, primary, tertiary: ")
    default = input("¿Tiene crédito en default? (default): yes, no: ")
    balance = float(input("Saldo promedio anual en euros (balance): "))
    housing = input("¿Tiene préstamo de vivienda? (housing): yes, no: ")
    loan = input("¿Tiene préstamo personal? (loan): yes, no: ")
    contact = input("Tipo de contacto (contact): unknown, telephone, cellular: ")
    day = int(input("Día del último contacto (day): "))
    month = input("Mes del último contacto (month): jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec: ")
    duration = int(input("Duración del último contacto en segundos (duration): "))
    campaign = int(input("Número de contactos en esta campaña (campaign): "))
    previous = int(input("Número de contactos antes de esta campaña (previous): "))
    poutcome = input("Resultado de la campaña anterior (poutcome): unknown, other, failure, success: ")

    # Convertir variables categóricas en dummies
    job_options = ["blue-collar", "pink-collar", "white-collar"]
    marital_options = ["married", "single"]
    education_options = ["primary", "secondary", "tertiary"]
    contact_options = ["cellular", "unknown"]
    poutcome_options = ["failure", "other", "success"]

    # Crear array de características y dummies en orden correcto (sin pdays)
    features = [
        age,
        1 if default == "yes" else 0,
        balance,
        1 if housing == "yes" else 0,
        1 if loan == "yes" else 0,
        day,
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(month) + 1,
        duration,
        campaign,
        previous,
    ]
    
    # Agregar dummies de job
    features.extend([1 if job == option else 0 for option in job_options])
    
    # Agregar dummies de marital
    features.extend([1 if marital == option else 0 for option in marital_options])

    # Agregar dummies de education
    features.extend([1 if education == option else 0 for option in education_options])

    # Agregar dummies de contact
    features.extend([1 if contact == option else 0 for option in contact_options])

    # Agregar dummies de poutcome
    features.extend([1 if poutcome == option else 0 for option in poutcome_options])

    # Convertir a numpy array y retornar
    return np.array([features])

# Obtener entrada del usuario y mostrarla
#nuevo_cliente = get_user_input()
nuevo_cliente = [
    45,  # age: Edad del cliente (45 años)
    0,   # default: ¿Tiene crédito en default? (0 = no)
    2000, # balance: Saldo promedio anual en euros (2000)
    1,   # housing: ¿Tiene préstamo de vivienda? (1 = sí)
    0,   # loan: ¿Tiene préstamo personal? (0 = no)
    10,  # day: Día del último contacto (10)
    7,   # month: Mes del último contacto (julio)
    120, # duration: Duración del último contacto en segundos (120)
    3,   # campaign: Número de contactos en esta campaña (3)
    1,   # previous: Número de contactos antes de esta campaña (1)
    
    # job dummies
    0,   # job_blue-collar: No es "blue-collar"
    0,   # job_pink-collar: No es "pink-collar"
    1,   # job_white-collar: Sí es "white-collar"
    
    # marital dummies
    1,   # marital_married: Casado
    0,   # marital_single: No es soltero
    
    # education dummies
    0,   # education_primary: No tiene educación primaria
    1,   # education_secondary: Tiene educación secundaria
    0,   # education_tertiary: No tiene educación terciaria
    
    # contact dummies
    0,   # contact_cellular: No fue contactado por celular
    1,   # contact_unknown: Contacto desconocido
    
    # poutcome dummies
    0,   # poutcome_failure: No fue un fracaso en la campaña anterior
    1,   # poutcome_other: Otro resultado en la campaña anterior
    0    # poutcome_success: No fue un éxito en la campaña anterior
]
print("Características del nuevo cliente (sin escalar):", nuevo_cliente)

# Escalar el nuevo cliente usando el scaler ajustado en los datos de entrenamiento
scaler_instance = scaler()  # Cargar el scaler
nuevo_cliente_scaled = scaler_instance.transform(nuevo_cliente)

# Realizar la predicción
prediccion = model_imp.predict(nuevo_cliente_scaled)

# Clasificar la predicción (0 o 1)
resultado = (prediccion > 0.5).astype("int32")

# Imprimir el resultado
print(f"Probabilidad de conversión: {prediccion[0][0]:.2f}")
print(f"Conversión predicha (1 = sí, 0 = no): {resultado[0][0]}")
