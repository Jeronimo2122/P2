import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Función para cargar y ajustar el scaler con los datos de entrenamiento
def cargar_scaler(X_train):
    scaler = StandardScaler()
   
    scaler.fit_transform(X_train)
    return scaler

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model2.h5')
X_train = np.load("X_train.npy")  # Cargar los datos de entrenamiento

# Datos de un cliente nuevo (sin `pdays`), ajustados al formato esperado
nuevo_cliente = [
    40,  # age: Edad del cliente (45 años)
    0,   # default: ¿Tiene crédito en default? (0 = no)
    1500, # balance: Saldo promedio anual en euros (2000)
    1,   # housing: ¿Tiene préstamo de vivienda? (1 = sí)
    0,   # loan: ¿Tiene préstamo personal? (0 = no)
    15,  # day: Día del último contacto (10)
    5,   # month: Mes del último contacto (julio)
    300, # duration: Duración del último contacto en segundos (120)
    2,   # campaign: Número de contactos en esta campaña (3)
    0,   # previous: Número de contactos antes de esta campaña (1)
    
    # job dummies
    1,   # job_blue-collar: No es "blue-collar"
    0,   # job_pink-collar: No es "pink-collar"
    0,   # job_white-collar: Sí es "white-collar"
    
    # marital dummies
    0,   # marital_married: Casado
    0,   # marital_single: No es soltero
    
    # education dummies
    1,   # education_primary: No tiene educación primaria
    0,   # education_secondary: Tiene educación secundaria
    0,   # education_tertiary: No tiene educación terciaria
    
    # contact dummies
    1,   # contact_cellular: No fue contactado por celular
    0,   # contact_unknown: Contacto desconocido
    
    # poutcome dummies
    0,   # poutcome_failure: No fue un fracaso en la campaña anterior
    1,   # poutcome_other: Otro resultado en la campaña anterior
    0    # poutcome_success: No fue un éxito en la campaña anterior
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

# Imprimir el resultado
print(f"Probabilidad de conversión: {prediccion[0][0]:.2f}")
print(f"Conversión predicha (1 = sí, 0 = no): {resultado[0][0]}")
