import cv2
import numpy as np
import time
import requests
from alert import Alert

# Configuración de la URL del backend
backend_url = "http://192.168.0.10:8080/api/alertas"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

last_detection_time = time.time()
alert = Alert("sound/alarm.wav")
critical_zone = (100, 50, 300, 350)
alert_count = 0  # Contador de activaciones de la alarma

# Método para enviar la imagen de la alerta
def send_alert_image(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': img_encoded.tobytes()}
    response = requests.post(f"{backend_url}/fotoSospechoso", files=files)
    print("Imagen enviada al backend") if response.ok else print("Error al enviar imagen")

# Función para activar la alarma
def activar_alarma():
    response = requests.put(f"{backend_url}/activarAlarma")
    print(response.text)

# Función para apagar la alarma
def apagar_alarma():
    response = requests.put(f"{backend_url}/apagarAlarma")
    print(response.text)

# Función para obtener el estado de las alertas
def obtener_estado():
    response = requests.get(f"{backend_url}/estado")
    if response.ok:
        print(response.json())
    else:
        print("Error al obtener el estado")

# Función para incrementar el contador de alertas
def incrementar_alerta():
    global alert_count
    alert_count += 1
    response = requests.post(f"{backend_url}/movimiento", json={"count": alert_count})
    print(response.text)

# Función para obtener las alertas
def obtener_alertas():
    response = requests.get(f"{backend_url}/obtenerAlertas")
    if response.ok:
        print(response.json())
    else:
        print("Error al obtener las alertas")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    cv2.rectangle(frame, (critical_zone[0], critical_zone[1]), 
                  (critical_zone[0] + critical_zone[2], critical_zone[1] + critical_zone[3]), 
                  (0, 255, 0), 2)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    movement_detected_in_zone = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if (x + w > critical_zone[0] and x < critical_zone[0] + critical_zone[2]) and \
           (y + h > critical_zone[1] and y < critical_zone[1] + critical_zone[3]):

            face_area_movement = fgmask[y:y + h, x:x + w]
            movement_detected_in_zone = np.sum(face_area_movement) > 5000

    if movement_detected_in_zone:
        current_time = time.time()
        if current_time - last_detection_time > 3:
            print("¡Alerta! Movimiento sospechoso detectado.")
            alert.play_alert()
            alert_count += 1
            send_alert_image(frame)
            requests.post(f"{backend_url}/increment-alert-count", json={"count": alert_count})
            last_detection_time = current_time

    cv2.imshow("Detección de rostro y movimiento", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
