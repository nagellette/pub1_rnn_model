from osgeo import gdal
import numpy as np
import batch_feeder as bf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
import support_functions as sf
from matplotlib import pyplot as plt
from sklearn import preprocessing

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
             "_bejing_full_projected_labels.tif"]

## raster values definition list
file_list_types = ["sar",
                   "sar",
                   "sar",
                   "sar",
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
    print("Mean: {0:2.2f}, Median {1:2.2f}, Std: {2:2.2f}, Min: {3:2.2f}, Max: {4:2.2f}".format(
        np.mean(raster_np), np.median(raster_np), np.std(raster_np), np.min(raster_np), np.max(raster_np)))
    if index != len(file_list) - 1:
        print(file)
        raster_np = (raster_np - np.mean(raster_np)) / np.std(raster_np)

    rasters_geo.append(raster_geo)
    rasters_np.append(raster_np)


## define project defaults TODO: move to seperate file and get from the file, preferably jso

## sub image size
SUB_IMAGE_COLS = 20
SUB_IMAGE_ROWS = 20

## shape of original images
IMAGE_COLS = rasters_np[0].shape[0]
IMAGE_ROWS = rasters_np[0].shape[1]

shuffle = True
skip_last_batch = True
batch_size = 64
train_bands = len(file_list) - 1
training_epochs = 20
learning_rate = 0.001
n_classes = SUB_IMAGE_ROWS * SUB_IMAGE_COLS

## model definition
## TODO: Autoencoder/Decoder shape created by array sizes, should be re-visited according to original definition if that differes
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(SUB_IMAGE_COLS, SUB_IMAGE_ROWS, train_bands)))
model.add(Conv2D(64, (3, 3), strides=(1, 1),
                 activation='relu'))
model.add(Conv2D(128, (3, 3), strides=(1, 1),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(3200, activation='relu'))
model.add(Dense(3200, activation='relu'))
model.add(Dense(3200, activation='relu'))
model.add(Dense(1600, activation='relu'))
model.add(Dense(1600, activation='relu'))
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
print(model.summary())

## creating feeder object
batch_feeder = bf.batch_feeder(IMAGE_COLS, IMAGE_ROWS, SUB_IMAGE_COLS, SUB_IMAGE_ROWS)


def generate_input_arrays(b_size):
    while True:
        for i in range(0, b_size):
            feed = batch_feeder.get_next_interval()

            for j in range(0, train_bands + 1):
                if j == 0:
                    temp_all_array = rasters_np[j]
                    temp = temp_all_array[feed[0]:feed[2], feed[1]:feed[3]]
                    xx = temp.flatten()
                elif j == train_bands:
                    temp_all_array = rasters_np[j]
                    yy = temp_all_array[feed[0]:feed[2], feed[1]:feed[3]].flatten()
                else:
                    temp_all_array = rasters_np[j]
                    temp = temp_all_array[feed[0]:feed[2], feed[1]:feed[3]]
                    xx = np.concatenate((xx, temp.flatten()))

            # create the initial batch or merge the batch with new value
            if i == 0:
                x = xx
                y = yy / 255.0
            else:
                x = np.concatenate((x, xx))
                y = np.concatenate((y, yy))

        # reshape the data and labels to batch count of rows as flattened images.
        x = np.reshape(x, (-1, SUB_IMAGE_ROWS * SUB_IMAGE_COLS * train_bands))
        y = np.reshape(y, (-1, SUB_IMAGE_ROWS * SUB_IMAGE_COLS))

        # convert the images to batches of layers
        yield x.reshape(x.shape[0], SUB_IMAGE_COLS, SUB_IMAGE_ROWS, train_bands), \
              y.reshape(y.shape[0], SUB_IMAGE_ROWS * SUB_IMAGE_COLS)


model.fit_generator(generate_input_arrays(batch_size),
                    steps_per_epoch=sf.math_support_functions.round_to_floor(batch_feeder.get_total_train_data(),
                                                                             batch_size),
                    epochs=training_epochs,
                    verbose=1,
                    callbacks=[history])

plt.plot(range(1, training_epochs + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
