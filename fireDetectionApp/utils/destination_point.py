import math

def get_destination_point(latitude, longitude, bearing, distance, radius):
    angular_distance = distance / radius

    nextLatitude = math.asin(
        math.sin(latitude) * math.cos(angular_distance) + 
        math.cos(latitude) * math.sin(angular_distance) * math.cos(bearing)
    )

    nextLongitude =  longitude + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(latitude),
        math.cos(angular_distance) - math.sin(latitude) * math.sin(nextLatitude)
    )

    return (
        math.degrees(nextLatitude),
        math.degrees(nextLongitude)
    )
