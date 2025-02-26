import cv2
import numpy as np

# Definir o caminho do vídeo (troque por 0 para usar a webcam)
video_path = "Livro.mp4"
cap = cv2.VideoCapture(video_path)  # cv2.VideoCapture(0) para webcam

# Definir os intervalos de cor HSV para azul
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Se não conseguiu ler um quadro, encerra o loop

    # Converter para HSV
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Criar uma máscara para capturar tons de azul
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # Exibir os quadros originais e a máscara
    cv2.imshow("Video Original", frame)
    cv2.imshow("Mascara Azul", mask_blue)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
