import os
import pandas as pd
import random
import numpy as np

import keras

from keras.models import Sequential
from keras.layers import Dense

from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

dir_path = "newexport1/"
dirs = os.listdir(dir_path)
file_count = len(dirs)

BATCH_SIZE = 5


def generate_data():
    while True:
        x = []
        y = []
        count = 0
        while count < BATCH_SIZE:
            file_id = random.randint(0, file_count - 1)
            date_id = random.randint(200, 500)

            stock_file = dirs[file_id]
            df = pd.read_csv(dir_path + stock_file, encoding='gbk', header=None)
            line_count = df.shape[0]
            if (date_id + 2) * 48 > line_count:
                continue

            stock_prices = df[2]

            # print(stock_file)
            # print(date_id)

            history_stock_price = stock_prices[(date_id - 200) * 48: date_id * 48]
            # print(history_stock_price)

            x.append(history_stock_price)

            tomorrow_stock_price = stock_prices[date_id * 48]
            after_tomorrow_stock_price = stock_prices[(date_id + 1) * 48]
            # print("tomorrow_stock_price")
            # print(tomorrow_stock_price)
            # print("after_tomorrow_stock_price")
            # print(after_tomorrow_stock_price)
            y.append((after_tomorrow_stock_price - tomorrow_stock_price) / tomorrow_stock_price)
            count += 1
        yield (np.array(x), np.array(y))


model = Sequential()
model.add(Dense(8192, input_dim=9600))
model.add(LeakyReLU())

model.add(Dense(4096))
model.add(LeakyReLU())

model.add(Dense(2048))
model.add(LeakyReLU())

model.add(Dense(1024))
model.add(LeakyReLU())

model.add(Dense(512))
model.add(LeakyReLU())

model.add(Dense(256))
model.add(LeakyReLU())

model.add(Dense(128))
model.add(LeakyReLU())

model.add(Dense(64))
model.add(LeakyReLU())

model.add(Dense(32))
model.add(LeakyReLU())

model.add(Dense(16))
model.add(LeakyReLU())

model.add(Dense(8))
model.add(LeakyReLU())

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1e-5, momentum=0.9))

callbacks_list = [ReduceLROnPlateau(), EarlyStopping(patience=30)]
model.fit_generator(generate_data(), steps_per_epoch=100, validation_data=generate_data(), validation_steps=30,
                    epochs=100)
model.save('./my_transfer.h5')
