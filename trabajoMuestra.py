import torch
import requests
import cv2
import numpy as np
import time
import base64
from alert import Alert

backend_url = "http://localhost:8080/api/alertas"

# Cargar el modelo YOLOv5m
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
last_detection_time = time.time()
alert = Alert("sound/alarm.wav")
critical_zone = (100, 50, 300, 350)

# Variables para seguimiento de movimiento
previous_positions = {}
movement_threshold = 20  # Umbral de movimiento para considerar que alguien está caminando o corriendo

def enviar_notificacion(tipo, mensaje, frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Crear el diccionario con los datos requeridos
    notificacion_data = {
        "tipo": tipo,  # Agregar el tipo
        "mensaje": mensaje,  # Agregar el mensaje
        "contenido": img_base64  # Usar "contenido" para la imagen
    }
    
    response = requests.post(f"{backend_url}/notificacion", json=notificacion_data)
    print("Notificación enviada:", response.status_code, response.json())

def enviar_alarma():
    alarma_data = {"sonidoActivado": True, "fechaActivacion": time.strftime("%Y-%m-%dT%H:%M:%S")}
    response = requests.post(f"{backend_url}/alarma", json=alarma_data)
    print("Alarma enviada:", response.status_code)

def enviar_imagen(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    imagen_base64 = base64.b64encode(img_encoded).decode('utf-8')
    imagen_data = {
        "contenido": imagen_base64,
        "fechaCaptura": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    response = requests.post(f"{backend_url}/imagen", json=imagen_data)
    print("Imagen enviada:", response.status_code)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    for detection in results.pred[0]:
        x1, y1, x2, y2, conf, cls = detection
        if model.names[int(cls)] == 'person':
            person_id = int(cls)  # Usamos la clase como ID para la persona
            
            # Calcular la posición actual
            current_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Centro de la persona
            
            # Dibujar rectángulo alrededor de la persona detectada
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if (int(x1) < critical_zone[0] + critical_zone[2] and int(x2) > critical_zone[0]) and \
               (int(y1) < critical_zone[1] + critical_zone[3] and int(y2) > critical_zone[1]):
                
                # Evaluar movimiento
                if person_id in previous_positions:
                    previous_position = previous_positions[person_id]
                    distance_moved = np.linalg.norm(np.array(current_position) - np.array(previous_position))
                    
                    # Solo activar si el movimiento es significativo
                    if distance_moved > movement_threshold:
                        print("¡Alerta! Movimiento sospechoso detectado en la zona.")
                        alert.play_alert()
                        enviar_notificacion("Movimiento", "Movimiento sospechoso detectado", frame)
                        enviar_alarma()
                        enviar_imagen(frame)

                # Actualizar la posición anterior
                previous_positions[person_id] = current_position

    cv2.imshow("Detección de personas y movimiento", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()