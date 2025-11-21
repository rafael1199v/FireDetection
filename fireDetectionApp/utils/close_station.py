import math
from typing import List
from fireDetectionApp.domain.fire_station import FireStation
from fireDetectionApp.utils.haversine_distance import calculate_haversine_distance

def get_close_station(latitudeStation: float, longitudeStation: float, stations: List[FireStation]) -> FireStation:

    min_distance = math.inf
    selected_station = None

    print(f"Analisis de puntos cercanos. {latitudeStation}, {longitudeStation}")

    for station in stations:
        distance =calculate_haversine_distance(
            firstLatitude=latitudeStation,
            firstLongitude=longitudeStation,
            secondLatitude=station.latitude,
            secondLongitude=station.longitude,
            radius=6_371_000
        )

        print(f"Estacion: {station.name} a una distancia: {distance}")
        print(f"Latidude: {station.latitude}, {station.longitude}")

        if distance < min_distance:
            selected_station = station
            min_distance = distance

    return selected_station
        

    