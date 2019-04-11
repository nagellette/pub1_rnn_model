from osgeo import gdal
import numpy as np
import batch_feeder as bf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential

## input file list
file_list = ["SAR_5m_amplitude_vv.tif",
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
             "probe_aggregated_speed_variance.tif",
             "_bejing_reduced_projected_labels.tif"]

## raster values definition list
file_list_types = ["reflectance",
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
                   "variance",
                   "label"]


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

'''
define project defaults TODO: move to seperete file and get from the file, preferably json.
'''

## sub image size
SUB_IMAGE_COLS = 80
SUB_IMAGE_ROWS = 80

## shape of original images
IMAGE_COLS = rasters_np[0].shape[0]
IMAGE_ROWS = rasters_np[0].shape[1]

shuffle = True
skip_last_batch = True
batch_size = 64
train_bands = len(file_list) - 1
training_epochs = 10
learning_rate = 0.001
n_classes = SUB_IMAGE_ROWS * SUB_IMAGE_COLS

## model definition
## TODO: Autoencoder/Decoder shape created by array sizes, should be re-visited according to original definition if that differes
model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=(SUB_IMAGE_COLS, SUB_IMAGE_ROWS, train_bands)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (4, 4), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (4, 4), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1600, activation='relu'))
model.add(Dense(1600, activation='relu'))
model.add(Dense(3200, activation='relu'))
model.add(Dense(n_classes, activation='relu'))

## loss function chosen as MSE to obtain values between 0-1
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])


## accuracy visualisation class
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

## creating feeder object
batch_feeder = bf.batch_feeder(IMAGE_COLS, IMAGE_ROWS, SUB_IMAGE_COLS, SUB_IMAGE_ROWS, shuffle)

def generate_input_arrays(batch_s):
    for i in range(0, batch_s):
        feed = batch_feeder.get_next_interval()

        if feed is not None:
            for j in range(0, train_bands + 1):
                if j ==0:
                    x = rasters_np[j].flatten()
                elif j == train_bands:
                    temp_all_array = rasters_np[j]
                    y = temp_all_array[feed[0]:feed[3]][feed[1]:feed[2]]
                else:
                    temp_all_array = rasters_np[j]
                    temp = temp_all_array[feed[0]:feed[3], feed[1]:feed[2]]
                    print(temp.shape)
                    np.concatenate((x, temp.flatten()))

            yield x.reshape(x.shape[0], SUB_IMAGE_COLS, SUB_IMAGE_ROWS, train_bands), y

model.fit_generator(generate_input_arrays(batch_size),
                    steps_per_epoch=batch_feeder.get_total_train_data(),
                    epochs=training_epochs,
                    verbose=1,
                    callbacks=[history])  # ,
# validation_data=(x_test, y_test))

plt.plot(range(1, training_epochs), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
