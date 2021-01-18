import ee
import json

def Polygons_to_MultiPolygon(Polygons):
    Polygons = list(filter(None, Polygons))
    MultiPoligon = {}
    properties = ["training", "test", "validation"]
    features = []
    for n, polygons in enumerate(Polygons):
        multipoligon = []
        for polygon in polygons.get('features'):
            multipoligon.append(polygon.get('geometry').get('coordinates'))
            
        features.append({
            "type": "Feature",
            "properties": {"name": properties[n]},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates":  multipoligon
            }
        }
        ) 
        
    MultiPoligon = {
        "geojson": {
            "type": "FeatureCollection", 
            "features": features
        }
    }

    return MultiPoligon

def GeoJSONs_to_FeatureCollections(geostore):
    feature_collections = []
    for n in range(len(geostore.get('geojson').get('features'))):
        # Make a list of Features
        features = []
        for i in range(len(geostore.get('geojson').get('features')[n].get('geometry').get('coordinates'))):
            features.append(
                ee.Feature(
                    ee.Geometry.Polygon(
                        geostore.get('geojson').get('features')[n].get('geometry').get('coordinates')[i]
                    )
                )
            )
            
        # Create a FeatureCollection from the list.
        feature_collections.append(ee.FeatureCollection(features))
    return feature_collections

def get_geojson_string(geom):
    coords = geom.get('coordinates', None)
    if coords and not any(isinstance(i, list) for i in coords[0]):
        geom['coordinates'] = [coords]
    feat_col = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}
    return json.dumps(feat_col)

def check_status_data(task, file_paths):
    status_list = list(map(lambda x: str(x), task.list()[:len(file_paths)])) 
    status_list = list(map(lambda x: x[x.find("(")+1:x.find(")")], status_list))
    
    return status_list

