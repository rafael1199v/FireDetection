import math


def calculate_haversine_distance(firstLatitude: float, firstLongitude: float, secondLatitude: float, secondLongitude: float, radius: float):
    """
    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1-a) )
    d = R ⋅ c
    """

    firstLatitude = math.radians(firstLatitude)
    firstLongitude = math.radians(firstLongitude)

    secondLatitude = math.radians(secondLatitude)
    secondLongitude = math.radians(secondLongitude)

    deltaLatitude = (secondLatitude - firstLatitude)
    deltaLongitude = (secondLongitude - firstLongitude)

    a = math.sin(deltaLatitude / 2) * math.sin(deltaLatitude / 2) + math.cos(firstLatitude) * math.cos(secondLatitude) * math.sin(deltaLongitude / 2) * math.sin(deltaLongitude / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = radius * c

    return distance
