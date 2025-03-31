import cv2
import numpy as np 

from .distance import fast_thinning, calculate_angle, find_distance

# Parámetros para detección de cambios de dirección
ANGLE_THRESHOLD = 30  # Umbral en grados para considerar cambio drástico
MIN_SEGMENT_LENGTH = 20  # Longitud mínima de un segmento
WINDOW_SIZE = 10  # Tamaño de la ventana para análisis
STEP_SIZE = 5  # Paso para el análisis de ventanas

def get_binary_img(img_class):
    img_green = img_class[:, :, 1]
    img_red = img_class[:, :, 2]
    img_blue = img_class[:, :, 0]

    green_mask = (img_green > img_red) & (img_green > img_blue)
    binary_img = np.uint8(green_mask) * 255
    return binary_img

def get_componets(binary_img):
    binary_img = cv2.medianBlur(binary_img, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Conectividad
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, 8)
    valid_components = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 150]
    return valid_components, labels

def get_segments(valid_components, labels):
    segments = []
    for component in valid_components:
        mask = (labels == component).astype(np.uint8)
        skeleton = fast_thinning(mask)
            
        # Obtener puntos ordenados del esqueleto
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
        for contour in contours:
            contour = contour.reshape(-1, 2)
            num_points = contour.shape[0]
            if num_points < WINDOW_SIZE:
                continue
                
            current_segment = []
            prev_direction = None
                
            # Procesar ventanas deslizantes
            for i in range(0, num_points - WINDOW_SIZE + 1, STEP_SIZE):
                window = contour[i:i+WINDOW_SIZE]
                x_win = window[:, 0]
                y_win = window[:, 1]
                 
                # Regresión lineal
                try:
                    coeffs = np.polyfit(x_win, y_win, 1)
                    m = coeffs[0]
                    dx, dy = 1, m
                    length = np.hypot(dx, dy)
                    current_direction = (dx/length, dy/length)
                except:
                    # Fallback a puntos extremos
                    dx = x_win[-1] - x_win[0]
                    dy = y_win[-1] - y_win[0]
                    length = np.hypot(dx, dy)
                    if length == 0:
                        continue
                    current_direction = (dx/length, dy/length)
                    
                # Si es el primer punto del segmento
                if not current_segment:
                    current_segment.extend(window.tolist())
                    prev_direction = current_direction
                    continue
                 
                # Calcular ángulo entre direcciones
                angle = calculate_angle(prev_direction, current_direction)
                    
                # Si el ángulo es menor que el umbral, continuamos el segmento actual
                if angle < ANGLE_THRESHOLD:
                    current_segment.extend(window[STEP_SIZE:].tolist())  # Evitar duplicados
                    prev_direction = current_direction  # Actualizamos con la última dirección
                else:
                   # Si el segmento actual es suficientemente largo, lo guardamos
                    if len(current_segment) >= MIN_SEGMENT_LENGTH:
                        segments.append(np.array(current_segment))
                    # Iniciamos nuevo segmento
                    current_segment = window.tolist()
                    prev_direction = current_direction
                
                # Añadir el último segmento si es válido
            if len(current_segment) >= MIN_SEGMENT_LENGTH:
                segments.append(np.array(current_segment))
    
    return segments

