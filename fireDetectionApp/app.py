from fireDetectionApp.gee import randomPoints, gee
from fireDetectionApp.utils import points_around, create_map
from pathlib import Path
import pickle
import pandas as pd

MODEL_FILE_NAME = "randomForestModel_v3"
MODEL_PATH = Path(f"saved_models/{MODEL_FILE_NAME}")
EARTH_RADIUS = 6_371_000

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def fire_detection_app():
    rf_model = load_model()
    points_sc = randomPoints.generar_puntos_aleatorios_santa_cruz(n_puntos=3)
    points_full_matrix = [points_around.generate_points_around(central_point=point,radius_m=EARTH_RADIUS, pointsNumber=5, distance_m=300) for point in points_sc]
    
    print("Obteniendo datos de GEE...")
    points_list = []
    for points_group in points_full_matrix:
        for point in points_group:
            lat, lon = point
            points_list.append((lat, lon))
    
    data_list = gee.get_batch_fire_points(points_list)
    
    print("Realizando predicciones...")
    df = pd.DataFrame(data_list)
    
    pred = rf_model.predict(df.drop(columns=['point_id', 'lon', 'lat']))
    pred_probabilites = rf_model.predict_proba(df.drop(columns=['point_id', 'lon', 'lat']))
    
    df["prediction"] = pred
    df["prediction_prob"] = pred_probabilites[:, 1]
    
    punto_actual = 0
    imagenes_generadas = []
    
    for idx, points_group in enumerate(points_full_matrix, 1):
        print(f"\n{'='*70}")
        print(f"PROCESANDO GRUPO #{idx} ({len(points_group)} puntos)")
        print(f"{'='*70}")
        
        n_points = len(points_group)
        probabilidades = df.iloc[punto_actual:punto_actual + n_points]['prediction_prob'].values
        
        for i, (point, prob) in enumerate(zip(points_group, probabilidades)):
            lat, lon = point
            estado = "ALERTA" if prob >= 0.60 else "✓ Normal"
            print(f"  Punto {i+1}: Lat={lat:.5f}, Lon={lon:.5f} | "
                  f"Prob={prob:.1%} | {estado}" f"Probabilidad real={prob}")
        
        nombre_archivo = f'mapa_calor_grupo_{idx}.png'
        archivo_guardado = create_map.crear_imagen_mapa_calor(
            points=points_group,
            probabilidades=probabilidades,
            nombre_archivo=nombre_archivo,
            grupo_num=idx
        )
        
        imagenes_generadas.append(archivo_guardado)
        print(f"\n✓ Imagen guardada: {archivo_guardado}")
        
        punto_actual += n_points
    
    print(f"\n{'='*70}")
    print("PROCESO COMPLETADO")
    print(f"Total de imágenes generadas: {len(imagenes_generadas)}")
    for img in imagenes_generadas:
        print(f"  - {img}")
    print(f"{'='*70}\n")
    
    return df, imagenes_generadas