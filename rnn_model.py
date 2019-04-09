from osgeo import gdal
import numpy as np
import batch_feeder as bf

## input file list
file_list = ["_bejing_reduced_projected_labels.tif",
             "SAR_5m_amplitude_vv.tif",
             "SAR_5m_amplitude_vh.tif",
             "SAR_5m_intensity_vv.tif",
             "SAR_5m_intensity_vh.tif",
             "sentinel_AOT_5m.tif",
             "sentinel_B02_5m.tif",
             "sentinel_B03_5m.tif",
             "sentinel_B04_5m.tif",
             "sentinel_B08_5m.tif",
             "sentinel_WVP_5m.tif",
             "probe_aggregated_trajectory_count.tif",
             "probe_aggregated_speed_avg.tif",
             "probe_aggregated_speed_stddev.tif",
             "probe_aggregated_speed_variance.tif"]

## raster values definition list
file_list_types =["label",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "reflectance",
                  "count",
                  "average",
                  "stddev",
                  "variance"]


## raster reader function TODO: add coordinate system getters
def read_raster(file_path):
    raster_file = gdal.Open(file_path)
    raster_np = raster_file.GetRasterBand(1).ReadAsArray()
    raster_geo = raster_file.GetGeoTransform()

    # close the raster file to free the memory
    raster_file = None

    # process input values
    # replace NaNs with zero
    raster_np[np.isnan(raster_np)] = 0.0

    # replace no data values with zero
    raster_np[raster_np == -9999.0] = 0.0

    # convert data type to float
    raster_np = raster_np.astype(np.float64)

    # return raster as numpy array and geo transformation parameters
    return raster_np, raster_geo


## empty list for
rasters_np = []
rasters_geo = []

## iterate over file list and read the rasterfiles
for index, file in enumerate(file_list):
    raster_np, raster_geo = read_raster("./data/" + file)
    print("Reading file: " + file)

    rasters_geo.append(raster_geo)
    rasters_np.append(raster_np)


## define project defaults TODO: move to seperete file and get from the file, preferably json.
SUB_IMAGE_COLS = 120
SUB_IMAGE_ROWS = 120

IMAGE_COLS = rasters_np[0].shape[0]
IMAGE_ROWS = rasters_np[0].shape[1]

batch_feeder = bf.batch_feeder(IMAGE_COLS, IMAGE_ROWS, SUB_IMAGE_COLS, SUB_IMAGE_ROWS, True)

print(batch_feeder.get_next_interval())
