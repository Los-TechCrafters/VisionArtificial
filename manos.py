import cv2
import mediapipe as mp

# Configuración para Mediapipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar detección de manos
    results = hands.process(image)

    # Convertir de nuevo a BGR para mostrar con OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dibujar puntos clave y conexiones en las manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Cambiar el color de los puntos y las conexiones
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Color verde para las conexiones
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)   # Color magenta para los puntos
            )

    # Mostrar la imagen resultante
    cv2.imshow("Detección de manos en tiempo real", image)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
