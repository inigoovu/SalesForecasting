import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers.core import Dense, Activation, Dropout, Flatten



from sklearn.preprocessing import MinMaxScaler

from keras import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import json

ids_locales = [24325,24675,20117,11537,21066,21276,20116,8406,16708,15150]

# IMPORT DATASET
df_all = pd.read_csv('../../Data/Ventas.csv',delimiter=';')

df_weather = pd.read_csv('../../Data/weather_timeserie.csv',delimiter=';')
file = '../../Data/meteo.txt'
with open(file) as train_file:
    dict_file = json.load(train_file)

# converting json dataset from dictionary to dataframe
df_weather2 = pd.DataFrame.from_dict(dict_file)
df_weather2.reset_index(level=0, inplace=True)

df_weather = df_weather[['Fecha','Clima']]
df_weather.columns = ['SalesDate','Desc_Weather']
df_weather2 = df_weather2[['fecha','prec','tmed','sol']]
df_weather2.columns = ['SalesDate','Rain','Temp_med','Sunny']
df_weather2 = df_weather2.applymap(lambda x: str(x.replace(',','.')))

for x in ids_locales:

    df_filter = df_all[df_all['StoreID'] == x]

    df_filter = df_filter.drop(['StoreID','DIF_INVENT','TI_JE_TOT','TI_TARJ_JE','V_BRUTA',
                                'V_EFEC_JE','V_TARJ_JE','V_SODEXHO','V_BORESA'],axis=1)

    df_filter = df_filter.fillna(0)


    df_filter['SalesDate'] = pd.to_datetime(df_filter['SalesDate'])



    df_filter = df_filter[df_filter['SalesDate']>= pd.to_datetime('2017-01-01')]
    df_filter = df_filter[df_filter['SalesDate'] <= pd.to_datetime('2019-10-12')]

    df_filter['SalesDate'] = pd.to_datetime(df_filter['SalesDate'])
    df_weather['SalesDate'] = pd.to_datetime(df_weather['SalesDate'])
    df_weather2['SalesDate'] = pd.to_datetime(df_weather2['SalesDate'])
    df_filter = pd.merge(df_filter, df_weather, on='SalesDate', how='inner')
    df_filter = pd.merge(df_filter, df_weather2, on='SalesDate', how='inner')

    cols = df_filter.select_dtypes(exclude=['float']).columns[1:]

    df_filter[cols] = df_filter[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    df_filter = df_filter.loc[:, df_filter.isin([' ', 'NULL', 0]).mean() < .2]


    df_filter = df_filter.fillna(0)



    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    dataset = df_filter.iloc[:,1:]

    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()

    # values[:, 0] = encoder.fit_transform(values[:, 0])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 30, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[11, 12, 13, 14, 15, 16, 17, 18, 19]], axis=1, inplace=True)
    # reframed.drop(reframed.columns[len(reframed.columns)-(len(dataset.columns)-1)], axis=1, inplace=True)

    reframed = reframed.iloc[:,:-(len(dataset.columns)-1)]

    values = reframed.values
    n_train_days = int(len(dataset) * 0.8)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))



    model = Sequential()

    model.add(LSTM(units=100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=20))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))




    from keras.optimizers import adam
    opt = adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=opt)


    history = model.fit(train_X, train_y, epochs=200, batch_size=1, validation_data=(test_X, test_y),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=40)], verbose=1, shuffle=False)


    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast

    inv_yhat = np.concatenate((yhat, test_X[:, 1:(len(dataset.columns))]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:(len(dataset.columns))]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    xhat = model.predict(train_X)
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    # invert scaling for forecast
    inv_xhat = np.concatenate((xhat, train_X[:, 1:(len(dataset.columns))]), axis=1)
    inv_xhat = scaler.inverse_transform(inv_xhat)
    inv_xhat = inv_xhat[:, 0]
    # invert scaling for actual
    train_y = train_y.reshape((len(train_y), 1))
    inv_x = np.concatenate((train_y, train_X[:, 1:(len(dataset.columns))]), axis=1)
    inv_x = scaler.inverse_transform(inv_x)
    inv_x = inv_x[:, 0]

    import sys

    orig_stdout = sys.stdout

    f = open('../../Results/errors_LSTM_' + str(x) + '.txt','w')

    sys.stdout = f


    print('Train Mean Absolute Error:', mean_absolute_error(inv_x, inv_xhat))
    print('Train Root Mean Squared Error:', np.sqrt(mean_squared_error(inv_x, inv_xhat)))
    print('Test Mean Absolute Error:', mean_absolute_error(inv_y, inv_yhat))
    print('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(inv_y, inv_yhat)))
    sys.stdout = orig_stdout
    f.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')

    plt.savefig('../../Results/loss_' + str(x) + '.png')

    aa = [x for x in range(len(inv_yhat))]
    plt.figure(figsize=(8, 4))
    plt.plot(aa, inv_y, marker='.', label="actual")
    plt.plot(aa, inv_yhat, 'r', label="prediction")
    # plt.tick_params(left=False, labelleft=True) #remove ticks
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Global_active_power', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.savefig('../../Results/result_'+str(x)+'.png')

    df_final = pd.DataFrame({'Fecha':df_filter.SalesDate.tail(len(inv_yhat)),'Real':inv_y.astype(int),'Prediccion':inv_yhat.astype(int)})
    df_final.to_csv('../../Results/V_NETA_'+str(x)+'.csv',sep=';')
    # plt.show();
