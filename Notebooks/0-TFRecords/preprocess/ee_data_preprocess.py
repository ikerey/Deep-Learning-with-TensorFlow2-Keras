import ee
import time
import json
import folium
from shapely.geometry import shape

from . import ee_collection_specifics
from .utils import Polygons_to_MultiPolygon, GeoJSONs_to_FeatureCollections, get_geojson_string, check_status_data

class Preprocess(object):
    """
    Training of Deep Learning models in Skydipper
    ----------
    privatekey_path: string
        A string specifying the direction of a json keyfile on your local filesystem
        e.g. "/Users/me/.privateKeys/key_with_bucket_permissions.json"
    """
    def __init__(self):
        #import env files & services auth
        self.ee_tiles = 'https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}'

    def composite(self, slugs=["Landsat-8-Surface-Reflectance"], init_date='2017-01-01', end_date='2017-12-31', lat=39.31, lon=0.302, zoom=6):
        """
        Returns a folium map with the composites.
        Parameters
        ----------
        slugs: list
            A list of dataset slugs to display on the map.
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.slugs = slugs
        self.init_date = init_date
        self.end_date= end_date
        self.lat = lat
        self.lon = lon
        self.zoom = zoom

        map = folium.Map(
                location=[self.lat, self.lon],
                zoom_start=self.zoom,
                tiles='OpenStreetMap',
                detect_retina=True,
                prefer_canvas=True
        )

        composites = []
        for n, slug in enumerate(self.slugs):
            composites.append(ee_collection_specifics.Composite(slug)(init_date, end_date))

            mapid = composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))

            folium.TileLayer(
            tiles=mapid['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            overlay=True,
            name=slug).add_to(map)

        self.composites = composites

        map.add_child(folium.LayerControl())
        return map

    def select_areas(self, attributes, zoom=6):
        """Create the geometries from which we will export the data.
        ----------
        attributes: list
            List of geojsons with the trainig, validation, and testing polygons.
        zoom: int
            A z-level for the map.
        """
        # Get MultiPolygon geostore object
        self.multi_polygon = Polygons_to_MultiPolygon(attributes)

        nFeatures = len(self.multi_polygon.get('geojson').get('features'))

        self.nPolygons = {}
        for n in range(nFeatures):
            multipoly_type = self.multi_polygon.get('geojson').get('features')[n].get('properties').get('name')
            self.nPolygons[multipoly_type] = len(self.multi_polygon.get('geojson').get('features')[n].get('geometry').get('coordinates'))
    
        for multipoly_type in self.nPolygons.keys():
            print(f'Number of {multipoly_type} polygons:', self.nPolygons[multipoly_type])
 
        # Returns a folium map with the polygons
        features = self.multi_polygon['geojson']['features']
        if len(features) > 0:
            shapely_geometry = [shape(feature['geometry']) for feature in features]
        else:
            shapely_geometry = None
    
        self.centroid = list(shapely_geometry[0].centroid.coords)[0][::-1]      

        map = folium.Map(location=self.centroid, zoom_start=zoom)

        if hasattr(self, 'composites'):
            for n, slug in enumerate(self.slugs):
                mapid = self.composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))

                folium.TileLayer(
                tiles=mapid['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                overlay=True,
                name=slug).add_to(map)

        nFeatures = len(features)
        colors = [['#FFFFFF', '#2BA4A0'],['#2BA4A0', '#FFE229'], ['#FFE229', '#FFFFFF']]
        for n in range(nFeatures):
            style_functions = [lambda x: {'fillOpacity': 0.0, 'weight': 4, 'color': color} for color in colors[n]]
            folium.GeoJson(data=get_geojson_string(features[n]['geometry']), style_function=style_functions[n],\
                 name=features[n].get('properties').get('name')).add_to(map)
        
        map.add_child(folium.LayerControl())
        return map

    def stack_images(self, feature_collections):
        """
        Stack the 2D images (input and output images of the Neural Network) 
        to create a single image from which samples can be taken
        """
        for n, slug in enumerate(self.slugs):
            # Stack RGB images
            if n == 0:
                self.image_stack = self.composites[n].visualize(**ee_collection_specifics.vizz_params_rgb(slug))
            else:
                #self.image_stack = ee.Image.cat([self.image_stack,self.composites[n].visualize(**ee_collection_specifics.vizz_params_rgb(slug))]).float()
                self.image_stack = ee.Image.cat([self.image_stack,self.composites[n]]).float()

        if self.kernel_size == 1:
            self.base_names = ['training_pixels', 'test_pixels']
            # Sample pixels
            vector = self.image_stack.sample(region = feature_collections[0], scale = self.scale,\
                                        numPixels=self.sample_size, tileScale=4, seed=999)

            # Add random column
            vector = vector.randomColumn(seed=999)

            # Partition the sample approximately 75%, 25%.
            self.training_dataset = vector.filter(ee.Filter.lt('random', 0.75))
            self.test_dataset = vector.filter(ee.Filter.gte('random', 0.75))

            # Training and validation size
            self.training_size = self.training_dataset.size().getInfo()
            self.test_size = self.test_dataset.size().getInfo()

        if self.kernel_size > 1:
            self.base_names = ['training_patches', 'test_patches']
            # Convert the image into an array image in which each pixel stores (kernel_size x kernel_size) patches of pixels for each band.
            list = ee.List.repeat(1, self.kernel_size)
            lists = ee.List.repeat(list, self.kernel_size)
            kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, lists)

            self.arrays = self.image_stack.neighborhoodToArray(kernel)

            # Training and test size
            nFeatures = len(self.multi_polygon.get('geojson').get('features'))
            nPolygons = {}
            for n in range(nFeatures):
                multipoly_type = self.multi_polygon.get('geojson').get('features')[n].get('properties').get('name')
                nPolygons[multipoly_type] = len(self.multi_polygon.get('geojson').get('features')[n].get('geometry').get('coordinates'))

            self.training_size = nPolygons['training']*self.sample_size
            self.test_size = nPolygons['test']*self.sample_size

    def start_TFRecords_task(self, feature_collections, feature_lists):
        """
        Create TFRecord's exportation task
        """

        # These numbers determined experimentally.
        nShards  = int(self.sample_size/20) # Number of shards in each polygon.

        if self.kernel_size == 1:
            # Export all the training validation and test data.   
            self.file_paths = []
            for n, dataset in enumerate([self.training_dataset, self.validation_dataset, self.test_dataset]):

                self.file_paths.append(self.bucket+ '/' + self.folder + '/' + self.base_names[n])

                # Create the tasks.
                task = ee.batch.Export.table.toCloudStorage(
                  collection = dataset,
                  description = 'Export '+self.base_names[n],
                  fileNamePrefix = self.folder + '/' + self.base_names[n],
                  bucket = self.bucket,
                  fileFormat = 'TFRecord',
                  selectors = list(self.image_stack.bandNames().getInfo())
                )

                task.start()

        if self.kernel_size > 1:
             # Export all the training validation and test data. (in many pieces), with one task per geometry.     
            self.file_paths = []
            for i, feature in enumerate(feature_collections):
                for g in range(feature.size().getInfo()):
                    geomSample = ee.FeatureCollection([])
                    for j in range(nShards):
                        sample = self.arrays.sample(
                            region = ee.Feature(feature_lists[i].get(g)).geometry(), 
                            scale = self.scale, 
                            numPixels = self.sample_size / nShards, # Size of the shard.
                            seed = j,
                            tileScale = 8
                        )
                        geomSample = geomSample.merge(sample)

                    desc = self.base_names[i] + '_g' + str(g)

                    self.file_paths.append(self.bucket+ '/' + self.folder + '/' + desc)

                    task = ee.batch.Export.table.toCloudStorage(
                        collection = geomSample,
                        description = desc, 
                        bucket = self.bucket, 
                        fileNamePrefix = self.folder + '/' + desc,
                        fileFormat = 'TFRecord',
                        selectors = list(self.image_stack.bandNames().getInfo())
                    )
                    task.start()

        return task

    def export_TFRecords(self, sample_size, kernel_size, scale, bucket, folder):
        """
        Export TFRecords to GCS.
        Parameters
        ----------
        sample_size: int
            Number of samples to extract from each polygon.
        kernel_size: int
            An integer specifying the height and width of the 2D images.
        scale: float
            Scale of the images in meters.
        bucket: string
            Bucket name.
        folder: string
            Folder path to save the data.
        """
        self.sample_size = sample_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.bucket = bucket
        self.folder = folder

        # Convert the GeoJSON to feature collections
        feature_collections = GeoJSONs_to_FeatureCollections(self.multi_polygon)
        
        # Convert the feature collections to lists for iteration.
        feature_lists = list(map(lambda x: x.toList(x.size()), feature_collections))

        ## Stack the 2D images to create a single image from which samples can be taken
        self.stack_images(feature_collections)

        ## Start task
        task = self.start_TFRecords_task(feature_collections, feature_lists)

        # Monitor task status
        print('Exporting TFRecords to GCS:')
        status_list = check_status_data(task, self.file_paths)
    
        while not status_list == ['COMPLETED'] * len(self.file_paths):
            status_list = check_status_data(task, self.file_paths)

            #Print temporal status 
            tmp_status = json.dumps(dict(zip(self.file_paths, status_list)))
            print('Temporal status: ', tmp_status)

            time.sleep(60)

        # Print final status
        print('Final status: COMPLETED')

            

