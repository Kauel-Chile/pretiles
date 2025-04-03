import zipfile
from pykml import parser
from lxml import etree
from shapely.geometry import Point, LineString, Polygon
import os

def read_kmz(file_path):
    """Lee un archivo KMZ y devuelve el objeto KML."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    with zipfile.ZipFile(file_path, 'r') as kmz:
        kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
        if not kml_files:
            raise ValueError("No se encontró ningún archivo KML dentro del KMZ")
        
        with kmz.open(kml_files[0]) as kml_file:
            kml_content = kml_file.read()
    
    return parser.fromstring(kml_content)

def extract_geometries(kml_obj):
    """Extrae todas las geometrías (Polygon, LineString) del KML."""
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    geometries = []
    kml_str = etree.tostring(kml_obj)
    root = etree.fromstring(kml_str)
    
    # Extraer Polygons
    for polygon in root.findall('.//kml:Polygon', namespaces=ns):
        outer = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces=ns)
        if outer is not None:
            coords = []
            for coord in outer.text.split():
                parts = coord.strip().split(',')
                if len(parts) >= 2:
                    lon, lat = map(float, parts[:2])
                    coords.append((lon, lat))
            if len(coords) >= 3:  # Mínimo 3 puntos para un polígono válido
                geometries.append(Polygon(coords))
    
    # Extraer LineStrings
    for linestring in root.findall('.//kml:LineString', namespaces=ns):
        coords_element = linestring.find('.//kml:coordinates', namespaces=ns)
        if coords_element is not None:
            coords = []
            for coord in coords_element.text.split():
                parts = coord.strip().split(',')
                if len(parts) >= 2:
                    lon, lat = map(float, parts[:2])
                    coords.append((lat, lon))
            if len(coords) >= 2:  # Mínimo 2 puntos para una línea válida
                geometries.append(LineString(coords))
    
    if not geometries:
        raise ValueError("No se encontraron geometrías válidas en el KML")
    
    return geometries

def check_coordinate(geometries, point, buffer_distance=0.001):
    """
    Verifica si un punto está dentro de un polígono o cerca de una línea.
    
    Args:
        geometries: Lista de geometrías (Polygon o LineString)
        point: Tupla con (longitud, latitud)
        buffer_distance: Radio de búsqueda para líneas (en grados decimales)
    
    Returns:
        bool: True si el punto está dentro o cerca
    """
    point_obj = Point(point)
    
    for geom in geometries:
        # Si es polígono, verifica contención
        if isinstance(geom, Polygon) and geom.contains(point_obj):
            return True
        # Si es línea, verifica proximidad
        elif isinstance(geom, LineString):
            buffered_line = geom.buffer(buffer_distance)
            if buffered_line.contains(point_obj):
                return True
    
    return False

def check_coordinates_in_kmz(kmz_path, coordinates_dict, buffer_distance=0.0001):
    """
    Verifica múltiples coordenadas contra un archivo KMZ.
    
    Args:
        kmz_path: Ruta al archivo KMZ
        coordinates_dict: Diccionario con {id: (lon, lat)}
        buffer_distance: Radio para líneas (en grados decimales)
    
    Returns:
        dict: {id: True/False} indicando qué coordenadas están contenidas
    """
    try:
        kml_obj = read_kmz(kmz_path)
        geometries = extract_geometries(kml_obj)
        print(geometries)
        results = {}
        for key, coord in coordinates_dict.items():
            results[key] = check_coordinate(geometries, coord, buffer_distance)
        
        return results
    
    except Exception as e:
        print(f"Error procesando el KMZ: {str(e)}")
        return {}
    
def check_pretil(kmz_file, data):
    keys = []
    for key, values in data.items():
        point = {'point': values[0][0]}
        results = check_coordinates_in_kmz(kmz_file, point)
        print(key,results)
        for _, is_inside in results.items():
            if is_inside:
                keys.append(key)
            
    return {key: data[key] for key in keys if key in data}
