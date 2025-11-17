from fireDetectionApp.gee import randomPoints, gee
from fireDetectionApp.utils import points_around
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
    points_sc = randomPoints.generar_puntos_aleatorios_santa_cruz(n_puntos=3)
    points_full_matrix = [points_around.generate_points_around(central_point=point,radius_m=EARTH_RADIUS, pointsNumber=5, distance_m=300) for point in points_sc]

    for points in points_full_matrix:
        for point in points:
            print(point)
            
        print()
    # points_list = [(point["lon"], point["lat"]) for point in points_sc]
    # data_list = gee.get_batch_fire_points(points_list)

    # rf_model = load_model()
    # df = pd.DataFrame(data_list)

    # pred = rf_model.predict(df.drop(columns=['point_id', 'lon', 'lat']))
    # pred_probabilites = rf_model.predict_proba(df.drop(columns=['point_id', 'lon', 'lat']))

    # df["prediction"] = pred
    # df["prediction_prob"] = pred_probabilites[:, 1]

    # print(df)

    # for index, row in df.iterrows():
    #     if row["prediction_prob"] >= 0.60:
    #         print(f"Se a detectado un incendio en el punto con lat: {row['lat']} y lon: {row["lon"]}!!!") 
    #     else:
    #         print(f"No se a detectado un incendio en el punto con lat: {row['lat']} y lon: {row["lon"]}") 
