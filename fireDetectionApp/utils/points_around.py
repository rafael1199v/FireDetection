import math
from fireDetectionApp.utils.destination_point import get_destination_point

def generate_points_around(central_point, radius_m, pointsNumber, distance_m):
    bearing_step = 360 / pointsNumber

    points = [(central_point["lat"], central_point["lon"])]

    for i in range(pointsNumber):
        bearing_deg = bearing_step * i
        bearing_rad = math.radians(bearing_deg)

        latitude_rad = math.radians(central_point["lat"])
        longitude_rad = math.radians(central_point["lon"])

        nextPoint = get_destination_point(
            latitude=latitude_rad,
            longitude=longitude_rad,
            bearing=bearing_rad,
            distance=distance_m,
            radius=radius_m
        )

        points.append(nextPoint)


    return points