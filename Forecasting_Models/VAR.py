import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statsmodels.stats.stattools import durbin_watson


import json
# 24325,24675,20117,11537,21066,21276,
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



uniques_weather = df_weather.Desc_Weather.unique()
res_dct = {}
for i in range(0,len(uniques_weather)):
    res_dct.update({uniques_weather[i]:i+1})

df_weather = df_weather.replace({"Desc_Weather": res_dct})


for x in ids_locales:

    df_filter = df_all[df_all['StoreID'] == x]
    # 20117
    # 21066


    df_filter['V_NETA'] = df_filter['V_NETA'].astype(float)

    df_filter = df_filter.drop(['StoreID','DIF_INVENT','TI_JE_TOT','TI_TARJ_JE','V_BRUTA',
                                'V_EFEC_JE','V_TARJ_JE','V_SODEXHO','V_BORESA'],axis=1)

    df_filter = df_filter.fillna(0)

    df_filter['SalesDate'] = pd.to_datetime(df_filter['SalesDate'])


    df_filter = df_filter[df_filter['SalesDate']>= pd.to_datetime('2017-01-01')]
    df_filter = df_filter[df_filter['SalesDate'] <= pd.to_datetime('2019-10-12')]


    df_weather['SalesDate'] = pd.to_datetime(df_weather['SalesDate'])
    df_weather2['SalesDate'] = pd.to_datetime(df_weather2['SalesDate'])
    df_filter = pd.merge(df_filter, df_weather, on='SalesDate', how='inner')
    df_filter = pd.merge(df_filter, df_weather2, on='SalesDate', how='inner')
    print(df_filter.head())


    cols = df_filter.select_dtypes(exclude=['float']).columns[1:]


    df_filter[cols] = df_filter[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    df_filter = df_filter.loc[:, df_filter.isin([' ', 'NULL', 0]).mean() < .1]

    print(df_filter.shape)
    df_filter = df_filter.fillna(0)
    print(df_filter.columns)

    pd.set_option('display.max_columns', None)
    import sys

    orig_stdout = sys.stdout

    f = open('../../Results/errors_forecasting_' + str(x) + '.txt', 'w')
    sys.stdout = f


    df_ventas = df_filter
    df_ventas_a = df_ventas


    maxlag=30
    test = 'ssr_chi2test'
    def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table
        are the P-Values. P-Values lesser than the significance level (0.05), implies
        the Null Hypothesis that the coefficients of the corresponding past values is
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        warnings.filterwarnings('ignore')
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        print(df)

    grangers_causation_matrix(df_ventas_a, variables = df_ventas_a.columns[1:])


    def cointegration_test(df, alpha=0.05,flag = 0):
        """Perform Johanson's Cointegration Test and Report Summary"""
        # if flag == 1:
        #     df = df+0.00001*abs(np.random.rand(852, 10))
            # df = df + 0.00001 * abs(np.random.rand(1479, 8))
        # print(df.head())
        out = coint_johansen(df,-1,5)
        # out = coint_johansen(df, 0, 12)
        d = {'0.90':0, '0.95':1, '0.99':2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1-alpha)]]
        def adjust(val, length= 6): return str(val).ljust(length)

        # Summary
        print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
        for col, trace, cvt in zip(df.columns, traces, cvts):
            print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)



    cointegration_test(df_ventas_a.iloc[:,1:],flag=1)


    nobs = 30
    df_ventas_train, df_ventas_test = df_ventas[0:-nobs], df_ventas[-nobs:]
    print(len(df_ventas_train))
    print(len(df_ventas_test))



    # CHECK STATIONARY

    def adfuller_test(series, signif=0.05, name='', verbose=False):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        r = adfuller(series, autolag='AIC')
        # r = adfuller(series,maxlag=30)
        output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
        p_value = output['pvalue']
        def adjust(val, length= 6): return str(val).ljust(length)

        # Print Summary
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key,val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
    #



    for name, column in df_ventas_train.iteritems():
        if name != 'SalesDate':
            adfuller_test(column, name=column.name)
            print('\n')



    model_ventas = VAR(df_ventas_train.iloc[:,1:])

    model_ventas_fitted = model_ventas.fit(30)


    out = durbin_watson(model_ventas_fitted.resid)
    def adjust(val, length= 6): return str(val).ljust(length)
    for col, val in zip(df_ventas_train.iloc[:,1:].columns, out):
        print(adjust(col), ':', round(val, 2))

    # Get the lag order
    lag_order = model_ventas_fitted.k_ar
    print('Lag order')
    print(lag_order)


    # Input data for forecasting
    forecast_input = df_ventas_train.iloc[:,1:].values[-lag_order:]

    fc = model_ventas_fitted.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=df_ventas_train.iloc[:,1:].index[-nobs:], columns=df_ventas_train.iloc[:,1:].columns )

    print(df_forecast.tail())
    print(df_ventas_test.tail())




    df_ventas_test = df_ventas_test.reset_index()
    df_results = df_forecast.reset_index(drop=True)

    df_results = df_results.rename(columns=lambda x: x+'_forecast')
    for index, row in df_results.iteritems():
        row[row < 0] = 0



    fig, axes = plt.subplots(nrows=int(len(df_ventas_train.iloc[:,1:].columns)/2), ncols=3, dpi=150, figsize=(10,10))
    for i, (col,ax) in enumerate(zip(df_ventas_train.iloc[:,1:].columns, axes.flatten())):
        df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        df_ventas_test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig('../../Results/variables_forecasting_' + str(x) + '.png')

    def forecast_accuracy(forecast, actual):
        mape = (np.mean(np.abs(forecast - actual)/np.abs(actual)))*100  # MAPE
        me = np.mean(forecast - actual)             # ME
        mae = np.mean(np.abs(forecast - actual))    # MAE
        mpe = np.mean((forecast - actual)/actual)   # MPE
        rmse = np.mean((forecast - actual)**2)**.5  # RMSE
        corr = np.corrcoef(forecast, actual)[0,1]   # corr
        mins = np.amin(np.hstack([forecast[:,None],
                                  actual[:,None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:,None],
                                  actual[:,None]]), axis=1)
        minmax = 1 - np.mean(mins/maxs)             # minmax
        return({'mape':mape, 'me':me, 'mae': mae,
                'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})



    for name, column in df_ventas_train.iteritems():
        if name != 'SalesDate':
            print(f'Forecast Accuracy of: {name}')

            df_final = pd.DataFrame({'Fecha': df_filter.SalesDate.tail(len(df_results[f'{name}_forecast'].values))
                                        , 'Real': df_ventas_test[name].values.astype(int),
                                     'Prediccion': df_results[f'{name}_forecast'].values.astype(int)})
            df_final.to_csv('../../Results/variables_' + str(name) + '_' + str(x) + '.csv', sep=';')

            accuracy_prod = forecast_accuracy(df_results[f'{name}_forecast'].values, df_ventas_test[name])
            for k, v in accuracy_prod.items():
                print(adjust(k), ': ', round(v,4))
    sys.stdout = orig_stdout
    f.close()
   
