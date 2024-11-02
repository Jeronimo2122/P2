import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Importar modelo entrenado

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
    pdays = int(input("Días desde el último contacto en la campaña anterior (pdays, -1 si no fue contactado): "))
    previous = int(input("Número de contactos antes de esta campaña (previous): "))
    poutcome = input("Resultado de la campaña anterior (poutcome): unknown, other, failure, success: ")

    # Convertir variables categóricas en dummies
    job_options = ["blue-collar", "pink-collar", "white-collar"]
    marital_options = ["married", "single"]
    education_options = ["primary", "secondary", "tertiary"]
    contact_options = ["cellular", "unknown"]
    poutcome_options = ["failure", "other", "success"]

    # Crear array de características y dummies en orden correcto
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
        pdays,
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
nuevo_cliente = get_user_input()
print("Características del nuevo cliente:", nuevo_cliente)

# Escalar los datos usando el mismo scaler que se usó en el entrenamiento
scaler = StandardScaler()
nuevo_cliente_scaled = scaler.transform(nuevo_cliente)

# Importar el modelo
model_imp = tf.keras.models.load_model('model2.h5')

# Realizar la predicción
prediccion = model_imp.predict(nuevo_cliente_scaled)

# Clasificar la predicción (0 o 1)
resultado = (prediccion > 0.5).astype("int32")

# Imprimir el resultado
print(f"Probabilidad de conversión: {prediccion[0][0]:.2f}")
print(f"Conversión predicha (1 = sí, 0 = no): {resultado[0][0]}")
