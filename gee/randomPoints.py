import ee
import random
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='config/.env')

ee.Authenticate()
ee.Initialize(project=os.getenv("PROJECT"))

def generar_puntos_aleatorios_santa_cruz(n_puntos=5, max_intentos=100):
  
    santa_cruz = ee.FeatureCollection("FAO/GAUL/2015/level1") \
        .filter(ee.Filter.eq('ADM0_NAME', 'Bolivia')) \
        .filter(ee.Filter.eq('ADM1_NAME', 'Santa Cruz')) \
        .geometry()
    
    bounds = santa_cruz.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0][0], bounds[0][1]
    max_lon, max_lat = bounds[2][0], bounds[2][1]
      
    puntos_validos = []
    intentos = 0
    
    while len(puntos_validos) < n_puntos and intentos < max_intentos:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        
        punto = ee.Geometry.Point([lon, lat])
        
        try:
           
            if santa_cruz.contains(punto, 1).getInfo():
                puntos_validos.append({
                    'id': f"punto_{len(puntos_validos) + 1}",
                    'lat': lat,
                    'lon': lon
                })
                print(f"https://www.google.com/maps?q={lat},{lon}")
            else:
                intentos += 1
        except Exception as e:
            print(f"Error: {e}")
            intentos += 1
    
    if len(puntos_validos) < n_puntos:
        print(f"Solo se encontraron {len(puntos_validos)} puntos válidos")
    else:
        print(f"Generados {len(puntos_validos)} puntos en Santa Cruz, Bolivia")
    
    return puntos_validos
