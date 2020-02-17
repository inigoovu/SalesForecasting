
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats



def create_corr_matrix(df, annot, size=(25, 25)):
    """
    Pearson correlation coefficient matrix.
    The Pearson correlation coefficient is a measure of the linear correlation between two variables.
    """
    # plt.clf()

    corr = df.corr(method='spearman')
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    if annot:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = plt.subplots(figsize=size)

    # sns.set(rc={'figure.figsize':(1,1)})
    plt.figure(figsize=(20, 10))
    fig = sns.heatmap(corr, mask=mask, square=False, cmap='RdYlGn', annot=annot,linecolor ='black',
                      cbar_kws={'label': 'Pearson correlation coefficient [-]'})

    fig.set_title('Correlation matrix')
    fig.tick_params(axis='x', rotation=90)
    fig.tick_params(axis='y', rotation=0)

    return fig




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



uniques_weather = df_weather.Desc_Weather.unique()
res_dct = {}
for i in range(0,len(uniques_weather)):
    res_dct.update({uniques_weather[i]:i+1})
print(res_dct)
df_weather = df_weather.replace({"Desc_Weather": res_dct})

# df_all = df_all.iloc[:82649,:]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
for x in ids_locales:

    df_filter = df_all[df_all['StoreID'] == x]

    df_filter['V_NETA'] = df_filter['V_NETA'].astype(float)

    df_filter = df_filter.drop(['StoreID', 'DIF_INVENT', 'TI_JE_TOT', 'TI_TARJ_JE', 'V_BRUTA',
                                'V_EFEC_JE', 'V_TARJ_JE', 'V_SODEXHO', 'V_BORESA'], axis=1)

    df_filter = df_filter.fillna(0)

    df_filter['SalesDate'] = pd.to_datetime(df_filter['SalesDate'])

    df_filter = df_filter[df_filter['SalesDate'] >= pd.to_datetime('2017-01-01')]
    df_filter = df_filter[df_filter['SalesDate'] <= pd.to_datetime('2019-10-12')]

    df_weather['SalesDate'] = pd.to_datetime(df_weather['SalesDate'])
    df_weather2['SalesDate'] = pd.to_datetime(df_weather2['SalesDate'])
    df_filter = pd.merge(df_filter, df_weather, on='SalesDate', how='inner')
    df_filter = pd.merge(df_filter, df_weather2, on='SalesDate', how='inner')

    from IPython.display import HTML
    h = HTML(df_filter.head().to_html())
    my_file = open('../../Results/table_'+str(x)+'.html', 'w')
    my_file.write(h.data)
    my_file.close()


    cols = df_filter.select_dtypes(exclude=['float']).columns[1:]

    df_filter[cols] = df_filter[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    df_filter = df_filter.fillna(0)

    df_filter['weekday'] = df_filter['SalesDate'].apply(lambda x: x.dayofweek)
    df_filter['month'] = df_filter['SalesDate'].apply(lambda x: x.month)
    df_filter['day'] = df_filter['SalesDate'].apply(lambda x: x.day)


    # https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/1.%20EDA%20%26%20Feature%20engineering.ipynb
    fig = create_corr_matrix(df_filter, True, size=(20,20))

    plt.show()




    for column in df_filter:
        if column != 'SalesDate':
            print(f'Statistics for {column}')
            stat, p = stats.normaltest(df_filter[column])
            print(f'Statistics={str(stat)}, p={str(p)}')
            alpha = 0.05
            if p > alpha:
                print('Data looks Gaussian (fail to reject H0')
            else:
                print('Data does not look Gaussian (reject H0)')

            sns.distplot(df_filter[column])
            plt.show()

            print(f'''Kurtosis of normal distribution: {stats.kurtosis(df_filter[column])}''')
            print(f'''Skewness of normal distribution: {stats.skew(df_filter[column])}''')


    df_filter = df_filter.loc[:,['SalesDate','V_NETA','year','quarter','month','day','week']]

    df_filter.sort_values('SalesDate',inplace=True, ascending=True)
    df_filter = df_filter.reset_index(drop=True)

    df_filter['year'] = df_filter['SalesDate'].apply(lambda x: x.year)
    df_filter['quarter'] = df_filter['SalesDate'].apply(lambda x: x.quarter)

    df_filter['weekday'] = df_filter['SalesDate'].apply(lambda x: x.dayofweek)
    # df_filter['weekday'] = df_filter.apply(lambda row: row['SalesDate'].weekday(), axis=1)
    df_filter['weekday'] = (df_filter['weekday'] < 5).astype(int)


    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.2)
    sns.boxplot(x='year',y='V_NETA',data=df_filter)
    plt.xlabel('year')
    plt.title('Box plot of Yearly V_NETA')
    sns.despine(left=True)
    plt.tight_layout()

    plt.subplot(1,2,2)
    sns.boxplot(x='quarter',y='V_NETA',data=df_filter)
    plt.xlabel('quarter')
    plt.title('Box plot of Quarterly V_NETA')
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()

    dic={0:'Weekend',1:'Weekday'}
    df_filter['Day'] = df_filter.weekday.map(dic)

    a = plt.figure(figsize=(9,4))
    plt1 = sns.boxplot('year','V_NETA',hue='Day',width=0.6,fliersize=3,data=df_filter)

    a.legend(loc='upper center',bbox_to_anchor=(0.5,1.00),shadow=True,ncol=2)
    sns.despine(left=True,bottom=True)
    plt.xlabel('')
    plt.tight_layout()
    plt.legend().set_visible(False)
    plt.show()

    plt1 = sns.factorplot('year','V_NETA',hue='Day',data=df_filter,size=4,aspect=1.5,legend=False)
    plt.title('Factor Plot of V_NETA by Weekend/Weekday')
    plt.tight_layout()
    sns.despine(left=True,bottom=True)
    plt.legend(loc='upper right')
    plt.show()


