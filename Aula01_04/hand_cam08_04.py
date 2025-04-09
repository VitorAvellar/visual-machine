import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Abrir a câmera
cap = cv2.VideoCapture(0)

# Cria uma imagem em branco para desenhar
drawing_canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao carregar o vídeo ou a câmera.")
        break

    # Redimensiona a imagem para 500x500
    frame = cv2.resize(frame, (500, 500))

    # Converte para RGB para o MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Se detectar a mão
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Desenha as landmarks da mão
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detecta a posição do dedo indicador (índice 8)
            index_finger_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            index_finger_y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            # Inicializa o canvas de desenho caso ainda não tenha sido criado
            if drawing_canvas is None:
                drawing_canvas = frame.copy()

            # Desenha no canvas com base na posição do dedo indicador
            cv2.circle(drawing_canvas, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)

    # Exibe o vídeo com as landmarks e o desenho
    cv2.imshow("Desenhando com a mão", drawing_canvas if drawing_canvas is not None else frame)

    # Aguarda para verificar se o usuário quer sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
