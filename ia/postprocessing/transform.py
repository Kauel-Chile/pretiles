import cv2
import pyproj
import numpy as np 

from collections import defaultdict
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
    for b, component in enumerate(valid_components):
        mask = (labels == component).astype(np.uint8)
        skeleton = fast_thinning(mask)
        
        # Verificar si el esqueleto está vacío
        if np.sum(skeleton) == 0:
            continue
            
        # Usar aproximación de contornos para reducir puntos redundantes
        contours, _ = cv2.findContours(
            skeleton, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_TC89_L1  # Algoritmo eficiente de aproximación
        )
        
        # Filtrar contornos pequeños inmediatamente
        valid_contours = [c for c in contours if len(c) >= WINDOW_SIZE]
        
        for i, contour in enumerate(valid_contours):
            contour = contour.reshape(-1, 2)
            num_points = contour.shape[0]
            
            current_segment = []
            prev_direction = None
            
            # Procesar ventanas deslizantes
            for j in range(0, num_points - WINDOW_SIZE + 1, STEP_SIZE):
                window = contour[j:j+WINDOW_SIZE]
                x_win = window[:, 0]
                y_win = window[:, 1]
                
                # Cálculo de dirección (regresión lineal o diferencia entre extremos)
                try:
                    coeffs = np.polyfit(x_win, y_win, 1)
                    m = coeffs[0]
                    dx, dy = 1, m
                except:
                    dx = x_win[-1] - x_win[0]
                    dy = y_win[-1] - y_win[0]
                
                length = np.hypot(dx, dy)
                if length == 0:
                    continue
                current_direction = (dx/length, dy/length)
                
                # Manejar segmento actual
                if not current_segment:
                    current_segment.extend(window.tolist())
                    prev_direction = current_direction
                    continue
                
                angle = calculate_angle(prev_direction, current_direction)
                if angle < ANGLE_THRESHOLD:
                    current_segment.extend(window[STEP_SIZE:].tolist())
                    prev_direction = current_direction
                else:
                    if len(current_segment) >= MIN_SEGMENT_LENGTH:
                        segments.append(np.array(current_segment))
                    current_segment = window.tolist()
                    prev_direction = current_direction
            
            # Añadir el último segmento del contorno
            if len(current_segment) >= MIN_SEGMENT_LENGTH:
                segments.append(np.array(current_segment))
    
    return segments

def get_measurement(segments, binary_mask, img_pretil):
    results = []

    # Procesar cada segmento por separado
    for segment_idx, segment in enumerate(segments):
        num_points = segment.shape[0]
         
        # Procesar puntos en el segmento con ventanas deslizantes
        for i in range(0, num_points - WINDOW_SIZE + 1, STEP_SIZE):
            window = segment[i:i+WINDOW_SIZE]
            x_win = window[:, 0]
            y_win = window[:, 1]
                
            # Regresión lineal para dirección
            try:
                coeffs = np.polyfit(x_win, y_win, 1)
                m = coeffs[0]
                dx, dy = 1, m
                length = np.hypot(dx, dy)
                dir_x, dir_y = dx/length, dy/length
            except:
                # Fallback a puntos extremos
                dx = x_win[-1] - x_win[0]
                dy = y_win[-1] - y_win[0]
                length = np.hypot(dx, dy)
                if length == 0:
                    continue
                dir_x, dir_y = dx/length, dy/length
                
            # Direcciones perpendiculares
            perp_up = (-dir_y, dir_x)
            perp_down = (dir_y, -dir_x)
                
            # Punto medio de la ventana
            mid_idx = i + WINDOW_SIZE // 2
            x0, y0 = segment[mid_idx]
                
            # Calcular distancias
            d_up = find_distance(x0, y0, perp_up, binary_mask)
            d_down = find_distance(x0, y0, perp_down, binary_mask)
            width = (d_up + d_down) * 0.25
                
            # Obtener altura del pretil
            height = img_pretil[y0, x0]
            results.append((x0, y0, segment_idx, width, height))
    
    return results

def get_coords(data, dem_info):
    result = defaultdict(list)
    coords_19s = [(get_grid_values(dem_info, x, y), idx, w_pretil, h_pretil) for x, y, idx, w_pretil, h_pretil in data]
    coords_wgs84 = [(transform_19s_wgs84(e, n), idx, w_pretil, h_pretil) for (e, n), idx, w_pretil, h_pretil in coords_19s]
    for coords, idx, w, h in coords_wgs84:
        result[idx].append((coords, w, h))
    return result

def get_grid_values(dem_info, w, h):
    grid_x, grid_y = dem_info.get('grid_x'), dem_info.get('grid_y')
    if not (0 <= w < grid_x.shape[1] and 0 <= h < grid_x.shape[0]):
        raise IndexError("Las coordenadas (w, h) están fuera de los límites de las matrices.")
    # Recuperar los valores de grid_x y grid_y en la posición (w, h)
    value_x = grid_x[h, w]
    value_y = grid_y[h, w]
    return value_x, value_y

def transform_19s_wgs84(coord_e, coord_n):
    source_crs = f"EPSG:32719"  # 19s   
    target_crs = "EPSG:4326"    # WGS84 for Google Maps
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    lon, lat = transformer.transform(coord_e, coord_n)
    return lat, lon
