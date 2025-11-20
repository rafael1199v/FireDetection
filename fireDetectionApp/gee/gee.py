import fireDetectionApp.gee_auth

import ee
import datetime

bolivia = ee.FeatureCollection("FAO/GAUL/2015/level0") \
            .filter(ee.Filter.eq('ADM0_NAME', 'Bolivia')) \
            .geometry()

def get_features_information(features):

    today = ee.Date(datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"))
    start_era5 = today.advance(-30, "day")

    era5_hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                    .filterDate(start_era5, today) \
                    .filterBounds(bolivia) \
                    .sort("system:time_start", False)

    size_era5 = era5_hourly.size().getInfo()
    print(f"Imágenes climáticas disponibles: {size_era5}")

    if size_era5 == 0:
        raise ValueError("No hay datos climáticos recientes para Bolivia.")

    clima_img = era5_hourly.first()

    def extract_weather(feature):
        point = feature.geometry()

        clima_reduced = clima_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(1000),
            scale=11132,
            maxPixels=1e9
        )

        tempC = ee.Number(clima_reduced.get("temperature_2m")).subtract(273.15)
        precip_mm = ee.Number(clima_reduced.get("total_precipitation")).multiply(1000)

        u_wind = ee.Number(clima_reduced.get("u_component_of_wind_10m", 0))
        v_wind = ee.Number(clima_reduced.get("v_component_of_wind_10m", 0))
        viento_ms = u_wind.hypot(v_wind)
        
        humedad_suelo = ee.Number(clima_reduced.get("volumetric_soil_water_layer_1"))

        return feature.set({
            "tempC": tempC,
            "precip_mm": precip_mm,
            "viento_ms": viento_ms,
            "humedad_suelo": humedad_suelo
        })
    

    features = features.map(extract_weather)

    for days_back in [5, 10, 15, 20, 30]:
        modis = ee.ImageCollection("MODIS/061/MOD09GA") \
                    .filterDate(today.advance(-days_back, "day"), today) \
                    .filterBounds(bolivia) \
                    .sort("system:time_start", False)

        size_modis = modis.size().getInfo()

        if size_modis > 0:
            print(f"Datos MODIS encontrados: {days_back} días atrás")
            modis_composite = modis.median()
            break
        else:
            print("No se encontraron datos MODIS, usando valores por defecto")
            modis_composite = None

    def extract_vegetation(feature):
        point = feature.geometry()

        if modis_composite is not None:
            ndvi = modis_composite.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
            nbr = modis_composite.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
            
            indices = ndvi.addBands(nbr).rename(['NDVI', 'NBR']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(500),
                scale=250,
                maxPixels=1e9,
                bestEffort=True
            )

            ndvi_val = ee.Number(indices.get("NDVI"))
            nbr_val = ee.Number(indices.get("NBR"))
        else:
            ndvi_val = ee.Number(0.3)
            nbr_val = ee.Number(0.2)

        return feature.set({
            'ndvi': ndvi_val,
            'nbr': nbr_val
        })

    features = features.map(extract_vegetation)

    srtm = ee.Image("USGS/SRTMGL1_003")

    def extract_elevation(feature):
        point = feature.geometry()

        elev = srtm.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(500),
            scale=30
        )

        return feature.set({
            'elevation_m': elev.get("elevation")
        })
    

    features = features.map(extract_elevation)
    features = features.getInfo()

    data_list = []

    for feature in features["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]

        data_list.append({
            'lon': coords[0],
            'lat': coords[1],
            'point_id': props.get('point_id'),
            'elevacion_m': props.get('elevation_m'),
            'humedad_suelo': props.get('humedad_suelo'),
            'nbr': props.get('nbr'),
            'ndvi': props.get('ndvi'),
            'precip_mm': props.get('precip_mm'),
            'tempC': props.get('tempC'),
            'viento_ms': props.get('viento_ms')
        })

    return data_list

def get_batch_fire_points(points_list):
    features = []

    index = 0

    for (lat, lon) in points_list:
        point = ee.Geometry.Point([lon, lat])
        features.append(ee.Feature(point, {'point_id': index}))
        index += 1

    points_fc = ee.FeatureCollection(features)
    data_list = get_features_information(points_fc)

    return data_list

