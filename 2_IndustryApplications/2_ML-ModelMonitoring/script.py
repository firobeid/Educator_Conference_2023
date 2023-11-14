#!/usr/bin/env python
# coding: utf-8
import numpy as np
import io
import sys
import os
import pandas as pd
import datetime
import gc #garabage collector
from io import BytesIO
import panel as pn
import holoviews as hv
import hvplot.pandas
import xlsxwriter
from warnings import filterwarnings
'''
development env: panel serve script.py --autoreload
prod prep: panel convert script.py --to pyodide-worker --out pyodide
'''

filterwarnings("ignore")
# hv.extension('bokeh')
pn.extension( "plotly", template="fast")

pn.state.template.param.update(
    # site_url="",
    site="ModelMonitor",
    title="Classification Model Metrics",
    # favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
#######################
###UTILITY FUNCTIONS###
#######################
def percentage(df):
    def segment(df):
        return round(df["Count"]/df["Count"].sum(),4)
    df["percent"] = segment(df)
    return df

def AUC(group):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(group['TARGET'],group['SCORE'])
    # N = sum(group["N"])
    N = round(len(group.loc[group["TARGET"].notna()]),0)
    cols = ["AUC","Count"]
    # return trapezoidal_rule(FPR.to_numpy(),TPR.to_numpy())
    return pd.Series([auc, N], index = cols)

def ROC(group):
    from sklearn.metrics import roc_curve
    FPR,TPR,T = roc_curve(group['TARGET'],group['SCORE'])
    cols = ['TPR', 'FPR']
    return pd.concat([pd.Series(TPR),pd.Series(FPR)], keys = cols, axis = 1)

def ks(group):
    from scipy.stats import ks_2samp
    y_real = group['TARGET']
    y_proba = group['SCORE']
    
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba
    
    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]
    
    ks_ = ks_2samp(class0['proba'], class1['proba'])
    
    N = round(len(group.loc[group["TARGET"].notna()]),0)
    cols = ["KS","Count"]
    
    return pd.Series([ks_[0], N], index = cols)

def psi(df):
    '''
    https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html#:~:text=To%20calculate%20the%20PSI%20we,the%20percents%20in%20each%20bucket.
    '''
    df[df == 0] = 0.001
    sub = df.copy()
    sub = sub.iloc[:,:-1].sub(df.validation,axis = 0)
    div = df.copy()
    div= div.iloc[:,:-1].div(df.validation, axis=0)
    div = np.log(div)
    return (sub*div).sum(axis = 0)

def add_extremes_OOT(df, name:str, score:str):
    '''
    Mitigate bias in OOT/Serving/baseline set that might not have high confidence scores or low confidence scores
    :param: name: str, name of the appid column
    :param: score: str, name of the score column
    '''
    # df.loc[len(df.index)] = [np.nan, "Extreme_Case_Max", np.nan, np.nan, np.nan,994.0,0.0009,np.nan,np.nan,np.nan,np.nan]
    # df.loc[len(df.index)] = [np.nan, "Extreme_Case_Min", np.nan, np.nan, np.nan,158.0,0.9999,np.nan,np.nan,np.nan,np.nan]
    df.loc[len(df.index)] = [np.nan for i in range(0,df.shape[1])]
    df.loc[(len(df.index)-1), [name, score]] = ["Extreme_Case_Max", 0.0009]
    df.loc[len(df.index)] = [np.nan for i in range(0,df.shape[1])]
    df.loc[(len(df.index)-1), [name, score]] = ["Extreme_Case_Min", 0.9999]
    return df

# def last_3months(df):
#     from datetime import datetime
#     from dateutil.relativedelta import relativedelta
#     from pandas.tseries.offsets import MonthEnd

#     end_of_month = ((pd.Timestamp(datetime.now().strftime('%Y-%m-%d')) - pd.Timedelta(70, unit='D')) + relativedelta(months=-1)) + MonthEnd(0)
#     start_of_month = end_of_month + MonthEnd(-3) + relativedelta(days=1)
#     end_of_month = end_of_month +relativedelta(hours=23, minutes=59, seconds=59)
#     print('Start Month %r --- End Month %r' % (start_of_month, end_of_month))
#     try:
#         date_column = list(filter(lambda x:x.endswith("DATE"),gains_df.columns))[0]
#     except:
#         date_column = 'CREATED_DATE'
#     return df[df[date_column].between(start_of_month, end_of_month)]

def gains_table_proba(data=None,target=None, prob=None):
    data = data.copy()
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['count'] = grouped.count()['target0']
    kstable['cum_total']=(kstable['count'] / kstable['count'].sum()).cumsum()
    kstable['events']  = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable['interval_rate'] = kstable['events'] / kstable['count']
    kstable = kstable.sort_values(by="min_prob", ascending=0).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['mid_point'] = np.nan
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 4) * 100

    #Formating
    kstable["cum_total"] = kstable["cum_total"].sort_values().values
    kstable = kstable.rename(columns={"min_prob":"low", "max_prob":"high"})
    kstable['mid_point'] = round((kstable['high'] + kstable['low']) / 2, 4)
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 15)
    # print(kstable)
    #Display KS
    from colorama import Fore
    ks_3mnths = "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0]))
    print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    kstable['cum_eventrate']= kstable['cum_eventrate'].str.replace("%","").astype(float)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].str.replace("%","").astype(float)
    kstable.index = list(range(10,0,-1))
    kstable = kstable.iloc[::-1]
    return(kstable, ks_3mnths)

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    # https://www.kaggle.com/code/podsyp/population-stability-index
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

    return round(10 **((158.313177 - UW5_Score) /274.360149), 18)

def lift_init(df:pd.DataFrame, baseline = None, is_baseline = True):
    from tqdm import tqdm
    # global standalone_scores_OOT
    cols = ['SCORE']
    
    lift_chart_data_OOT = pd.DataFrame()
    for q in tqdm([10,20,50,100]):
        # df_new["QUARTER"] = pd.PeriodIndex(df_new.CREATE_DATE, freq='Q')
        # fd = baseline.dropna(subset = period_metrics.value)[cols].apply(lambda col: pd.qcut(col.rank(method='first'),q = q, ), axis = 0).copy()
        # pd.cut(prod['SCORE'], bins = pd.qcut(baseline['SCORE'],10, retbins = True)[1]) 
        if is_baseline == True:
            # print(df)
            # print(df.dropna(subset = ['MONTHLY']))
            fd = df.dropna(subset = [period_metrics.value])[cols].apply(lambda col: pd.cut(col, bins = pd.qcut(col,q=q, retbins = True)[1]) , axis = 0).copy()
            fd = pd.concat([df.dropna(subset = [period_metrics.value])[period_metrics.value], df.dropna(subset = [period_metrics.value])['TARGET'], fd], axis = 1)
            fd = pd.concat([fd.groupby(x)['TARGET'].mean().fillna(0) for x in fd[cols]], axis = 1, keys = cols)
            fd.index.name = 'SCORE_BAND'
            
        else:
            # print(baseline.dropna(subset = [period_metrics.value])[cols].values.ravel().shape)
            # print(pd.qcut(baseline.dropna(subset = [period_metrics.value])[cols].values.ravel(),q=q, retbins = True))
            bins_ = pd.qcut(baseline.dropna(subset = [period_metrics.value])[cols].values.ravel(),q=q, retbins = True)[1]
            fd = df.groupby([period_metrics.value]).apply(lambda col: col[cols].apply(lambda col: pd.cut(col, bins = bins_), axis = 0)).copy()
            # fd = df.groupby(period_metrics.value).apply(lambda col: col[cols].apply(lambda col: pd.cut(col, bins = pd.qcut(col,q=q, retbins = True)[1]), axis = 0)).copy()
            fd = pd.concat([df[period_metrics.value], df['TARGET'], fd], axis = 1)
            fd = fd.groupby(period_metrics.value).apply(lambda col: pd.concat([col.groupby(x)['TARGET'].mean().fillna(0) for x in col[cols]], axis = 1, keys = cols))
            fd.index.names = [period_metrics.value, 'SCORE_BAND']
        # fd['APPLICATION_MONTH'] = fd['APPLICATION_MONTH'].astype(str)
        fd = fd.reset_index()
        fd['BINS'] = q
        lift_chart_data_OOT = lift_chart_data_OOT.append(fd)
    if is_baseline == True:
        lift_chart_data_OOT[period_metrics.value] = 'Baseline'
       
    standalone_scores_OOT = lift_chart_data_OOT.melt(id_vars=[period_metrics.value,'BINS','SCORE_BAND'],value_vars=cols,
                                        var_name='SCORE', 
                                        value_name='BAD_RATE').dropna().reset_index(drop = True).copy()
    standalone_scores_OOT[['BINS', 'SCORE_BAND']] = standalone_scores_OOT[['BINS', 'SCORE_BAND']].astype(str)
    standalone_scores_OOT = pd.concat([standalone_scores_OOT["BINS"] + "-" + standalone_scores_OOT["SCORE_BAND"] + "-" + standalone_scores_OOT["SCORE"], 
                                       standalone_scores_OOT[[period_metrics.value,'BAD_RATE']]], axis = 1).rename(columns = {0:'BINS_SCOREBAND_SCORE'})
    standalone_scores_OOT = standalone_scores_OOT.pivot(index = 'BINS_SCOREBAND_SCORE', columns=period_metrics.value)['BAD_RATE'].reset_index()
    standalone_scores_OOT.index.name = ""
    standalone_scores_OOT.columns.name = ""
    standalone_scores_OOT = pd.concat([standalone_scores_OOT['BINS_SCOREBAND_SCORE'].str.split('-', expand=True),
                                   standalone_scores_OOT],axis = 1).rename(columns ={0:'BINS', 1: 'SCORE_BAND', 2: 'SCORE'}).drop(columns = 'BINS_SCOREBAND_SCORE')
    # standalone_scores_OOT[['BINS', 'SCORE_BAND']] = standalone_scores_OOT[['BINS', 'SCORE_BAND']]#.astype(int)
    standalone_scores_OOT['BINS'] = standalone_scores_OOT['BINS']
    standalone_scores_OOT.sort_values(['SCORE', 'SCORE_BAND'], inplace = True)
    return standalone_scores_OOT, lift_chart_data_OOT

def lift_init_plots(df:pd.DataFrame, is_baseline = True):
    from tqdm import tqdm
    # global standalone_scores_OOT
    cols = ['SCORE']

    lift_chart_data_OOT = pd.DataFrame()
    for q in tqdm([10,20,50,100]):
        # df_new["QUARTER"] = pd.PeriodIndex(df_new.CREATE_DATE, freq='Q')
        # fd = baseline.dropna(subset = period_metrics.value)[cols].apply(lambda col: pd.qcut(col.rank(method='first'),q = q, ), axis = 0).copy()
        # pd.cut(prod['SCORE'], bins = pd.qcut(baseline['SCORE'],10, retbins = True)[1]) 
        # fd = df.dropna(subset = period_metrics.value)[cols].apply(lambda col: pd.cut(col, bins = pd.qcut(col,q=q, retbins = True)[1]) , axis = 0).copy()
        if is_baseline == True:
            fd = df.dropna(subset = period_metrics.value)[cols].apply(lambda col: pd.qcut(col.rank(method='first'),q = q, labels=range(1, q + 1)), axis = 0).copy()
            fd = pd.concat([df.dropna(subset = period_metrics.value)[period_metrics.value], df.dropna(subset = period_metrics.value)['TARGET'], fd], axis = 1)
            fd = pd.concat([fd.groupby(x)['TARGET'].mean().fillna(0) for x in fd[cols]], axis = 1, keys = cols)
            fd.index.name = 'SCORE_BAND'
            
        else:
            fd = df.groupby(period_metrics.value).apply(lambda col: col[cols].apply(lambda col: pd.qcut(col.rank(method='first'),q = q, labels=range(1,q + 1)), axis = 0)).copy()
            fd = pd.concat([df[period_metrics.value], df['TARGET'], fd], axis = 1)
            fd = fd.groupby(period_metrics.value).apply(lambda col: pd.concat([col.groupby(x)['TARGET'].mean().fillna(0) for x in col[cols]], axis = 1, keys = cols))
            # print(fd.index)
            fd.index.names = [period_metrics.value, 'SCORE_BAND']
            # fd = fd.reset_index(names = ['APPLICATION_MONTH', 'SCORE_BAND'])  
        fd = fd.reset_index()
        # fd['APPLICATION_MONTH'] = fd['APPLICATION_MONTH'].astype(str)
        fd['BINS'] = q
        lift_chart_data_OOT = lift_chart_data_OOT.append(fd)
    if is_baseline == True:
        lift_chart_data_OOT[period_metrics.value] = 'Baseline'
    lift_chart_data_OOT.sort_values(['SCORE', 'SCORE_BAND'], inplace = True)   
    standalone_scores_OOT = lift_chart_data_OOT.melt(id_vars=[period_metrics.value,'BINS','SCORE_BAND'],value_vars=cols,
                                        var_name='SCORE', 
                                        value_name='BAD_RATE').dropna().reset_index(drop = True).copy()
    standalone_scores_OOT[['BINS', 'SCORE_BAND']] = standalone_scores_OOT[['BINS', 'SCORE_BAND']].astype(str)
    standalone_scores_OOT = pd.concat([standalone_scores_OOT["BINS"] + "-" + standalone_scores_OOT["SCORE_BAND"] + "-" + standalone_scores_OOT["SCORE"], 
                                       standalone_scores_OOT[[period_metrics.value,'BAD_RATE']]], axis = 1).rename(columns = {0:'BINS_SCOREBAND_SCORE'})
    standalone_scores_OOT = standalone_scores_OOT.pivot(index = 'BINS_SCOREBAND_SCORE', columns=period_metrics.value)['BAD_RATE'].reset_index()
    standalone_scores_OOT.index.name = ""
    standalone_scores_OOT.columns.name = ""
    standalone_scores_OOT = pd.concat([standalone_scores_OOT['BINS_SCOREBAND_SCORE'].str.split('-', expand=True),
                                   standalone_scores_OOT],axis = 1).rename(columns ={0:'BINS', 1: 'SCORE_BAND', 2: 'SCORE'}).drop(columns = 'BINS_SCOREBAND_SCORE')
    standalone_scores_OOT[['BINS', 'SCORE_BAND']] = standalone_scores_OOT[['BINS', 'SCORE_BAND']].astype(int)
    standalone_scores_OOT['BINS'] = standalone_scores_OOT['BINS']
    standalone_scores_OOT.sort_values(['SCORE', 'SCORE_BAND'], inplace = True)
    return standalone_scores_OOT

def save_csv(df, metric):
    from io import StringIO
    sio = StringIO()
    df.to_csv(sio)
    sio.seek(0)
    return pn.widgets.FileDownload(sio, embed=True, filename='%s.csv'%metric)

def get_xlsx(df1,df2,df3,df4,df5,df6):
    from io import BytesIO
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, sheet_name="PSI")
    df2.to_excel(writer, sheet_name="AUC")
    df3.to_excel(writer, sheet_name="KS")
    df4.to_excel(writer, sheet_name="LABEL_DRIFT")
    df5.to_excel(writer, sheet_name="LABEL_Tables")
    df6.to_excel(writer, sheet_name="GAINS_Tables")
    writer.save() # Important!
    output.seek(0) # Important!
    return pn.widgets.FileDownload(output,embed=True, filename='results.csv', button_type="primary")


def expected_calibration_error(y, proba, bins = 'fd'):
  import numpy as np
  bin_count, bin_edges = np.histogram(proba, bins = bins)
  n_bins = len(bin_count)
  bin_edges[0] -= 1e-8 # because left edge is not included
  bin_id = np.digitize(proba, bin_edges, right = True) - 1
  bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins)
  bin_probasum = np.bincount(bin_id, weights = proba, minlength = n_bins)
  bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
  bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
  ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
  return ece, bin_probamean, bin_ymean, bin_id, bin_count, bin_edges
###############################
###END OFF UTILITY FUNCTIONS###
###############################

text = """
#Classification Model Metrics
## AUTHOR: [`FIRAS ALI OBEID`](https://www.linkedin.com/in/feras-obeid/) 
###  GNU General Public License v3.0 (GPL-3.0)
#### Developed while working at [OppFi Inc.](https://www.oppfi.com/)

This tool performs ML model ,in production, monitoring across time, 
where production weeks/months/quarters are compared too a selective baseline.

1. Upload a CSV containing:

**(Date)** Highly Recommended but **optional** 
**(Score)** Probability Predictions  
**(Target)** Binary Target/True Label
 
2. Check the box if you CSV has a DATE column, otherwise dates are generated based on current timestamp and spanning back by 
timedelta of csv length in hourly frequency.

3. Choose & press the right columns in the `Select Boxes` below when you upload a csv

4. Select a baseline date slice **mandatory**. If your baseline is from a different time then the production time,
make sure to append it to the csv before uploading.

5. Press Get Metrics

6. Wait few seconds and analyze the updated charts
"""



# date = str(input('What is the name off the date column: ').upper())
# id_ = str(input('What is the name off the APP name/ID column: ').upper())
# score = str(input('What is the name off the score column (i.e UW5,DM_QL...): ').upper())
# target = str(input('What is the name off the Target column (i.e Real target values such as PD70_RATIO...: ').upper())

file_input = pn.widgets.FileInput(align='center')
date_selector = pn.widgets.Select(name='Select Date Column',)
check_date = pn.widgets.Checkbox(name = '<--',value = False) # T/F
target_selector = pn.widgets.Select(name='Select Target Variable(True Label)')
score_selector = pn.widgets.Select(name='Select Predictions Column(Raw Probaility)')
period_metrics = pn.widgets.Select(name='Select Period', options = ['MONTHLY','WEEKLY', 'QUARTERLY'])

date_range_ = pn.widgets.DateRangeSlider(name='Baseline Period',) #value=(start, end), start=start, end=end

random_seed = pn.widgets.IntSlider(name='Random Seed for Random Generated Data (OnSet)', value=42, start=0, end=1000, step=1)

button = pn.widgets.Button(name='Get Metrics')
widgets = pn.WidgetBox(
    pn.panel(text, margin=(0, 20)),
    pn.panel('**Check box if your data has a date column *before uploading the file* \n (otherwise keep it empty)**'),
    check_date,
    file_input,
    random_seed,
    pn.panel('\n'),
    date_selector,
    target_selector,
    score_selector,
    period_metrics,
    date_range_,
    button
)

# start, end = stocks.index.min(), stocks.index.max()
# year = pn.widgets.DateRangeSlider(name='Year', value=(start, end), start=start, end=end)
# ,id_:'ID',


def get_data():
    global df
    if file_input.value is None:
        np.random.seed(random_seed.value)
        try:
            df = pd.DataFrame({'DATE': pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = 9999)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H"),
                            'ID': [i for i in range(10000)],
                            'SCORE':np.random.uniform(size = 10000),
                            'TARGET': np.random.choice([0,1],10000, p=[0.9,0.1])})
        except:
            df = pd.DataFrame({'DATE': pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = 9999 + 1)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H"),
                            'ID': [i for i in range(10000)],
                            'SCORE':np.random.uniform(size = 10000),
                            'TARGET': np.random.choice([0,1],10000, p=[0.9,0.1])})            
        # df.to_csv("test_upload.csv")
    else:
        df = BytesIO()
        df.write(file_input.value)
        df.seek(0)
        try:
                df = pd.read_csv(df, error_bad_lines=False).apply(pd.to_numeric, errors='ignore')
        except:
                df = pd.read_csv(df, error_bad_lines=False)

        df = df.select_dtypes(exclude=["category"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = [i.upper() for i in df.columns]
    return df

def update_target(event):
    df = get_data()
    cols = list(df.columns)
    date_selector.set_param(options=cols)
    target_selector.set_param(options=cols)
    score_selector.set_param(options=cols)
    # print(check_date.value)
    # print(type(df.DATE.min()))
    if check_date.value == True:
        date_column = [i.find("DATE") for i in df.columns] 
        date_column = [date_column.index(i) for i in [i for i in date_column if i !=-1]]
        if len(date_column) > 0:
            df = df.iloc[:,date_column].iloc[:,[0]]
            df.columns = ['DATE']
            print(type(df.DATE.min()))
            start, end = pd.Timestamp(df.DATE.min()),  pd.Timestamp(df.DATE.max())
            try:
                date_range_.set_param(value=(start, end), start=start, end=end)
            except:
                date_range_.set_param(value=(end, start), start=end, end=start)
        else:
            print('Creating synthetic dates')
            synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df))), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H") #remove len(df) - 1
            df['DATE'] = synthetic_date[:len(df)]
            start, end = df.DATE.min(), df.DATE.max()
            date_range_.set_param(value=(start, end), start=start, end=end)
    else:
        print('Creating synthetic dates')
        synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df))), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H")
        df['DATE'] = synthetic_date[:len(df)]
        start, end = df.DATE.min(), df.DATE.max()
        date_range_.set_param(value=(start, end), start=start, end=end)
file_input.param.watch(update_target, 'value')
update_target(None)

@pn.depends(button.param.clicks)
def run(_):
    print(random_seed.value)
    print(score_selector.value)
    df = get_data()
    try:
        if file_input.value is None:
            pass
        elif check_date.value == True:
            df = df.rename(columns={date_selector.value:'DATE',score_selector.value:'SCORE',target_selector.value:'TARGET'})
        else:
            synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df) - 1)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H")
            df['DATE'] = synthetic_date[:len(df)]
            df = df.rename(columns={score_selector.value:'SCORE',target_selector.value:'TARGET'})
    except Exception as e:
        return pn.pane.Markdown(f"""{e}""")
    try:
        df.DATE = pd.to_datetime(df.DATE, format="%Y-%m-%d %H:%M:%S", utc = True)
        # print(pd.to_datetime(df.DATE,utc = True))
        df["MONTHLY"] = df["DATE"].dt.strftime('%Y-%m')
        print(f"J - DAYS COUNT: {datetime.datetime.now() - pd.Timestamp('2023-03-06 03:27')}" )
        df['QUARTERLY'] = pd.PeriodIndex(df.DATE, freq='Q').astype(str)
        df['WEEKLY'] = pd.PeriodIndex(df.DATE, freq='W').astype(str)
    except Exception as e:
        return pn.pane.Markdown(f"""{e}""")    
    df = df.reset_index().rename(columns={df.index.name:'ID'}) #crate synthetic prediction ID for my code to run
    # df = df.dropna(subset = 'TARGET', axis = 1)
    df = df[~(df.TARGET.isna()) | (df.SCORE.isna())]
    if df.TARGET.nunique() > 2:
        df.TARGET = np.where(df.TARGET > 0 , 1 , 0)      
    df.SCORE = df.SCORE.astype(np.float64)

    

    # baselines
    # try:
    #     baseline = df.set_index('MONTHLY').loc[date_range_.value[0]: date_range_.value[1]].reset_index().copy()
    # except:
    #     baseline = df.copy()
    #     baseline = baseline.set_index('MONTHLY')
    #     baseline.index = pd.to_datetime(baseline.index)
    #     baseline = baseline.loc[date_range_.value[0]: date_range_.value[1]].reset_index()
    #     baseline["MONTHLY"] = baseline["MONTHLY"] .dt.strftime('%Y-%m')
    print(date_range_.value[0])
    print(date_range_.value[1])

    baseline = df.set_index('DATE').sort_index().loc[date_range_.value[0]: date_range_.value[1]].reset_index()
    print(baseline.DATE.min())
    print(baseline.DATE.max())
    print(df.DATE.max())

    # print(df.set_index('DATE').loc[date_range_.value[0]: date_range_.value[1]].index.max())
    #prods
    # prod = df.loc[~df.MONTHLY.isin(list(baseline.MONTHLY.unique()))].copy()
    prod_dates = df.set_index('DATE').sort_index().index.difference(baseline.set_index('DATE').index)
    # print(prod_dates)
    prod = df.set_index('DATE').loc[prod_dates].reset_index()
    if len(baseline) > len(prod):
        prod = baseline
    ##START##
    intiate = pn.pane.Alert('''### Baseline Period: \n%s to %s
    '''%(baseline.DATE.min(),baseline.DATE.max()), alert_type="info")
    intiate2 = pn.pane.Alert('''### Production Period: \n%s to %s
    '''%(prod.DATE.min(),prod.DATE.max()), alert_type="info")
    if prod.equals(baseline):
        intiate3 = pn.pane.Alert('''### Baseline Set is identical to Production Set \n Please choose a slice to be a baseline set''', alert_type="danger")
    else:
        intiate3 = None
    ##PSI##
    baseline_psi = baseline.copy()
    prod_psi = prod.copy()

    baseline_psi = add_extremes_OOT(baseline_psi, name = 'ID', score = 'SCORE')
    prod_psi["DEC_BANDS"] = pd.cut(prod_psi['SCORE'], bins = pd.qcut(baseline_psi['SCORE'],10, retbins = True)[1]) 
    prod_psi = prod_psi.groupby([period_metrics.value,
                                    "DEC_BANDS"]).agg(Count = ("DEC_BANDS",
                                    "count")).sort_index(level = 0).reset_index()
    prod_psi = prod_psi.groupby(period_metrics.value).apply(percentage).drop("Count",axis = 1)

    baseline_psi["DECILE"] = pd.cut(baseline_psi['SCORE'], bins = pd.qcut(baseline_psi['SCORE'],10, retbins = True)[1]) 
    baseline_psi = baseline_psi["DECILE"].value_counts()
    baseline_psi = baseline_psi / sum(baseline_psi)
    baseline_psi = baseline_psi.reset_index().rename(columns={"index":"DEC_BANDS", "DECILE": "percent"})
    baseline_psi[period_metrics.value] = "validation"
    baseline_psi = baseline_psi[[period_metrics.value, "DEC_BANDS", "percent"]]

    prod_psi = pd.concat([prod_psi,baseline_psi])

    prod_psi = prod_psi.pivot(index = "DEC_BANDS", columns=period_metrics.value)["percent"]
    if len(baseline) < len(prod):
        psi_ = psi(prod_psi).to_frame("%s_PSI"%period_metrics.value)
        psi_results = pn.widgets.DataFrame(psi_)
    else: 
        psi_ = pd.DataFrame()
        psi_results = pn.pane.Alert("### Choose a Baseline in the left banner to get PSI results", alert_type="warning")
    #CONFIGS
    baseline['QUARTERLY'] = 'Baseline: '+ baseline['QUARTERLY'].unique()[0] + '_' + baseline['QUARTERLY'].unique()[-1]
    baseline['MONTHLY'] = 'Baseline: '+ baseline['MONTHLY'].unique()[0] + '_' + baseline['MONTHLY'].unique()[-1]
    baseline['WEEKLY'] = 'Baseline: '+ baseline['WEEKLY'].unique()[0] + '_' + baseline['WEEKLY'].unique()[-1]
    #AUC
    auc_b = baseline.groupby([period_metrics.value]).apply(AUC)
    auc_p = prod.groupby([period_metrics.value]).apply(AUC)
    baseline_auc = pn.widgets.DataFrame(auc_b)
    prod_auc = pn.widgets.DataFrame(auc_p,name = 'AUC') #autosize_mode='fit_columns'
    
    from sklearn.metrics import roc_curve
    from holoviews import Slope
    b_label = baseline.MONTHLY.min()
    FPR,TPR,T = roc_curve(baseline['TARGET'],baseline['SCORE'])
    roc_baseline = pd.concat([pd.Series(TPR), pd.Series(FPR)], keys = ['TPR', 'FPR'], axis = 1)
    roc_baseline_p = roc_baseline.hvplot.line(x ='FPR', y = 'TPR', label = "Baseline", color = 'red')

    roc_plot = prod.groupby([period_metrics.value]).apply(ROC).reset_index(level = 0).hvplot.line(x ='FPR', y = 'TPR', title = "%s ROC (Production VS %s)"%(period_metrics.value, b_label),
                                                        groupby = period_metrics.value, width = 600, height = 500, label = "Prod",
                                                        xlim = (0,1), ylim = (0,1), grid = True) *  Slope(slope=1, y_intercept=0).opts(color='black', line_dash='dashed') * roc_baseline_p 
    #KS
    ks_b = baseline.groupby([period_metrics.value]).apply(ks)
    ks_p = prod.groupby([period_metrics.value]).apply(ks)
    baseline_ks = pn.widgets.DataFrame(ks_b)
    prod_ks = pn.widgets.DataFrame(ks_p,name = 'AUC') #autosize_mode='fit_columns'

    #LIFT
    baseline_lift_raw, baseline_lift_raw_bins = lift_init(df = baseline)
    baseline_lift_raw = baseline_lift_raw.rename(columns = {'Baseline': b_label})
    prod_lift_raw,  prod_lift_raw_bins = lift_init(df = prod, baseline = baseline, is_baseline = False)
    cols_b = baseline_lift_raw.columns.drop(['BINS', 'SCORE'])
    cols = prod_lift_raw.columns.drop(['BINS', 'SCORE'])

    baseline_lift = baseline_lift_raw.loc[baseline_lift_raw.BINS =='10',cols_b]
    prod_lift = prod_lift_raw.loc[prod_lift_raw.BINS =='10',cols]
    # prod_lift = pd.concat([prod_lift.dropna(subset = [col]).dropna(axis = 1).reset_index(drop = 1) for col in prod_lift][1:], axis = 1)
    lift_table = prod_lift_raw.loc[prod_lift_raw.BINS =='10',cols].melt(id_vars="SCORE_BAND", 
                                var_name='column', 
                                value_name='value').dropna().reset_index(drop = True).rename(columns = {'column':period_metrics.value , 'value': 'Target_PCT'})
    # print(prod_lift_raw_bins.loc[prod_lift_raw_bins.BINS ==10])
    lift_table = lift_table.hvplot.table(groupby = period_metrics.value, title="%s Lift Table"%period_metrics.value, hover = True, responsive=True, 
                shared_axes= False, fit_columns = True,
                padding=True , index_position = 0, fontscale = 1.5)
    # print(prod_lift_raw.loc[prod_lift_raw.BINS =='10',cols])
    # print(baseline_lift_raw.loc[baseline_lift_raw.BINS == '10',cols_b])
    prod_lift_raw['BINS'] = prod_lift_raw['BINS'].astype(int)
    baseline_lift_raw['BINS'] = baseline_lift_raw['BINS'].astype(int)
    
    prod_lift_raw_bins['SCORE_BAND'] = prod_lift_raw_bins['SCORE_BAND'].astype(str)
    # prod_lift_raw_bins['BINS'] = prod_lift_raw_bins['BINS'].astype(str)

    baseline_lift_raw_bins['SCORE_BAND'] = baseline_lift_raw_bins['SCORE_BAND'].astype(str)
    # baseline_lift_raw_bins['BINS'] = baseline_lift_raw_bins['BINS'].astype(str)  

    # print(prod_lift_raw.loc[:,list(cols)+['BINS']])
    p1 = prod_lift_raw_bins.set_index('SCORE_BAND'
                                                      ).reset_index().hvplot.line(x = 'SCORE_BAND', groupby = ['BINS', period_metrics.value],
                                                          grid = True, width = 1200, height = 500,
                                                          label = 'Production', rot = 45)
    
    # print(baseline_lift_raw_bins)  
    # print(prod_lift_raw_bins)                                                    
    p2 = prod_lift_raw_bins.set_index('SCORE_BAND'
                                                        ).reset_index().hvplot.scatter(x = 'SCORE_BAND', groupby = ['BINS', period_metrics.value], grid = True, color='DarkBlue', label='Production', rot = 45) 

    b_label = baseline.MONTHLY.min()
    # print(baseline_lift_raw.loc[baseline_lift_raw.BINS == '10',cols_b][b_label])
    b1 = baseline_lift_raw_bins.hvplot.line(x = 'SCORE_BAND', groupby = ['BINS'],
                                                            grid = True, width = 1200, height = 500,
                                                            line_dash='dashed', color = 'black', label = b_label, rot = 45)

    b2 = baseline_lift_raw_bins.hvplot.scatter(x = 'SCORE_BAND', groupby = ['BINS'], grid = True, color='DarkGreen', label = b_label, rot = 45) 

    final_lift_plots = (p1*p2*b1*b2).opts(ylabel = '%target_rate_mean', title = "%s Lift Chart " % (period_metrics.value.title())) 

    #LABEL_DRIFT
    mean_score_prod = prod.groupby(period_metrics.value).agg(MEAN_SCORE=("SCORE","mean"), MEAN_TARGET=("TARGET","mean"),Count = ("TARGET","count"))
    mean_score_base = baseline.groupby(period_metrics.value).agg(MEAN_SCORE=("SCORE","mean"), MEAN_TARGET=("TARGET","mean"),Count = ("TARGET","count"))
    baseline_label_drift = pn.widgets.DataFrame(mean_score_base)
    prod_label_drift = pn.widgets.DataFrame(mean_score_prod,name = 'DRIFT')

    #Lift Tables
    # gains_final_all,_ = gains_table_proba(prod,'TARGET', 'SCORE')
    lift_data = pd.concat([baseline_lift, prod_lift], axis = 0)
    lift_data = pd.concat([lift_data.dropna(subset = [col]).dropna(axis = 1).reset_index(drop = 1) for col in lift_data][1:], axis = 1).dropna(axis = 1, how = 'any')
    lift_data = lift_data.loc[:,~lift_data.columns.duplicated()].set_index('SCORE_BAND')
    if (lift_data.shape[1] > 4) | (lift_data.shape[0] > 10):
        prod_lift = pn.pane.Markdown('### Please download the csv as the lift table will congest the screen')
    else:
        prod_lift = pn.widgets.DataFrame(lift_data,name = 'LIFT')
    #GAINS_TABLE
    gains_final_prod,_ = gains_table_proba(prod,'TARGET', 'SCORE')
    gains_final_base,_ = gains_table_proba(baseline,'TARGET', 'SCORE')
    gains_final_base.index.names = [b_label]
    gains_final_p = pn.widgets.DataFrame(gains_final_prod.set_index(['low','high']),name = 'GAINS',)
    gains_final_b = pn.widgets.DataFrame(gains_final_base.set_index(['low','high']),name = 'GAINS',)

    ece, bin_probamean, bin_ymean, bin_id, bin_count, bin_edges = expected_calibration_error(prod.TARGET.values, prod.SCORE.values)
    error = pd.DataFrame(np.array([bin_probamean, bin_ymean]).T,columns= ["SCORE_MEAN", "TARGET_MEAN"])
    error_plot = error.hvplot.scatter(x ='SCORE_MEAN', y = 'TARGET_MEAN', width = 800, height = 500, label = "Bin (Score vs Target Mean)", title = 'Model Scores Calibration (--- Perfect Calibration)',
                                                        xlim = (0,1), ylim = (0,1), grid = True, xlabel = 'Bins Mean of Scores', ylabel = 'Bins Mean of Target') *  Slope(slope=1, y_intercept=0,legend = 'Perfect Calibration').opts(color='black', line_dash='dashed')  
    variable_ = pn.pane.Alert('''### FJ Day Count: \n%s 
    '''%(datetime.datetime.now() - pd.Timestamp('2023-03-06 03:27')), alert_type="success")
    return pn.Tabs(
        ('Metrics', pn.Column(
                    pn.Row(intiate, intiate2, intiate3, width = 1200),
                    '# PSI',
                    pn.Row(psi_results, save_csv(psi_, 'PSI')),
                    '# AUC',
                    pn.Row(prod_auc, baseline_auc, save_csv(pd.concat([auc_b, auc_p], axis = 0), 'AUC')),
                    '# KS',
                    pn.Row(prod_ks, baseline_ks, save_csv(pd.concat([ks_b, ks_p], axis = 0), 'KS')),
                    '# LABEL DRIFT',
                    pn.Row(prod_label_drift, baseline_label_drift, save_csv(pd.concat([mean_score_base, mean_score_prod], axis = 0), 'LABEL_DRIFT')),
                    '# LIFT TABLES',
                    pn.Row(prod_lift, save_csv(lift_data, 'LIFT_TABLES')),
                    '# GAINS TABLE',
                    pn.Row(gains_final_b, gains_final_p, save_csv(pd.concat([gains_final_base, gains_final_prod], axis = 1), 'GAINS_TABLES')),
                    get_xlsx(psi_, pd.concat([auc_b, auc_p], axis = 0), pd.concat([ks_b, ks_p], axis = 0), pd.concat([mean_score_base, mean_score_prod], axis = 0), lift_data, pd.concat([gains_final_base, gains_final_prod], axis = 1)), 
                    pn.Row(variable_, width = 200),
                             )
        ), #sizing_mode='stretch_width'
        ('Charts', pn.Column(pn.Row(roc_plot.opts(legend_position = 'bottom_right'), error_plot.opts(legend_position = 'top_left')) ,
                             lift_table,
                             final_lift_plots.opts(legend_position = 'bottom_right')
                            )
        )
    
    )
        

    # return pn.Tabs(
    #         ('Analysis', pn.Column(
    #             pn.Row(vol_ret, pn.layout.Spacer(width=20), pn.Column(div, table), sizing_mode='stretch_width'),
    #             pn.Column(pn.Row(year, investment), return_curve, sizing_mode='stretch_width'),
    #             sizing_mode='stretch_width')),
    #         ('Timeseries', timeseries),
    #         ('Log Return', pn.Column(
    #             '## Daily normalized log returns',
    #             'Width of distribution indicates volatility and center of distribution the mean daily return.',
    #             log_ret_hists,
    #             sizing_mode='stretch_width'
    #         ))
    #     )

pn.Row(pn.Column(widgets), pn.layout.Spacer(width=30), run).servable()













# Caveats
# The maximum sizes set in either Bokeh or Tornado refer to the maximum size of the message that 
# is transferred through the web socket connection, which is going to be larger than the actual 
# size of the uploaded file since the file content is encoded in a base64 string. So if you set a
# maximum size of 100 MB for your application, you should indicate your users that the upload
# limit is a value that is less than 100 MB.

# When a file whose size is larger than the limits is selected by a user, their browser/tab may
# just crash. Alternatively the web socket connection can close (sometimes with an error message
# printed in the browser console such as [bokeh] Lost websocket 0 connection, 1009 (message too 
# big)) which means the application will become unresponsive and needs to be refreshed.

# app = ...

# MAX_SIZE_MB = 150

# pn.serve(
#     app,
#     # Increase the maximum websocket message size allowed by Bokeh
#     websocket_max_message_size=MAX_SIZE_MB*1024*1014,
#     # Increase the maximum buffer size allowed by Tornado
#     http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1014}
# )