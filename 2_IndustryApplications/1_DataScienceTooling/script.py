import numpy as np
import io
import sys
import os
import pandas as pd

import gc #garabage collector
from io import BytesIO
import panel as pn
import holoviews as hv
import hvplot.pandas
from warnings import filterwarnings
'''
development env: panel serve script.py --autoreload
prod prep: panel convert script.py --to pyodide-worker --out pyodide
'''

filterwarnings("ignore")
hv.extension('bokeh')


text = """
#  Feature Distribution and Stats
## AUTHOR: [`FIRAS ALI OBEID`](https://www.linkedin.com/in/feras-obeid/) 
###  GNU General Public License v3.0 (GPL-3.0)
#### Developed while working at [OppFi Inc.](https://www.oppfi.com/)

This tool performs feature binning by equal intervals and by equal pouplations in each interval vs bad rate/target binary variable
To get the feature deep dive feature distribution:

1. Upload a CSV (only numerical data)

2. Choose & press on the binary (0 / 1) target column in the `Select Target Variable` section below

3. Press Run Analysis

4. Wait few seconds and analyze the updated charts
"""

file_input = pn.widgets.FileInput(align='center')
selector = pn.widgets.MultiSelect(name='Select Target Variable')
button = pn.widgets.Button(name='Run Analysis')
widgets = pn.WidgetBox(
    pn.panel(text, margin=(0, 10)),
    pn.panel('Upload a CSV containing  (X) features and  (y) binary variable:', margin=(0, 10)),
    file_input,
    selector,
    button
)


def closest(lst, K):
    try:
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    except:
        return K
control_max = lambda x: x.max() * 1.01 if x.max() > 0 else (x.max() * 0.99 if x.max() < 0 else x.max() + 0.01)
control_min = lambda x: x.min() * 0.99 if x.min() > 0 else (x.min() * 1.01 if x.min() < 0 else x.min() - 0.01)

def get_data():
    global target, New_Refit_routing
    if file_input.value is None:
        New_Refit_routing = pd.DataFrame({"Open_accounts": np.random.randint(1,50,100000),
                                          "Income": np.random.randint(1000,20000,100000),
                                          "Years_of_experience": np.random.randint(0,20,100000),
                                          "default": np.random.random_integers(0,1,100000)})
        target = "default"
    else:
        New_Refit_routing = BytesIO()
        New_Refit_routing.write(file_input.value)
        New_Refit_routing.seek(0)
        try:
            New_Refit_routing = pd.read_csv(New_Refit_routing, error_bad_lines=False).apply(pd.to_numeric, errors='ignore')#.set_index("id")
        except:
            New_Refit_routing = pd.read_csv(New_Refit_routing, error_bad_lines=False)
        target = None
        New_Refit_routing = New_Refit_routing.select_dtypes(exclude=['datetime', "category","object"])
        New_Refit_routing = New_Refit_routing.replace([np.inf, -np.inf], np.nan)
        # New_Refit_routing = New_Refit_routing[[cols for cols in New_Refit_routing.columns if New_Refit_routing[cols].nunique() >= 2]] #remove columns with less then 2 unique values
    return target, New_Refit_routing


def update_target(event):
    _ , New_Refit_routing = get_data()
    target = list(New_Refit_routing.columns)
    selector.set_param(options=target, value=target)

file_input.param.watch(update_target, 'value')
update_target(None)



def stats_():
    global stats
    stats = New_Refit_routing.describe().T
    stats["Missing_Values(%)"] = (New_Refit_routing.isna().sum() / len(New_Refit_routing)) * 100
    stats = pd.concat([stats, New_Refit_routing.quantile(q = [.01, .05, .95, .99]).T.rename(columns = {0.01: '1%', 0.05: '5%', 0.95: '95%', 0.99:'99%'})], axis = 1)
    stats = stats[['count', 'mean', 'std', 'min', '1%', '5%' ,'25%', '50%', '75%', '95%', '99%', 'max','Missing_Values(%)']]
    stats = stats.round(4).astype(str)

def cuts_(target):
    global test, test2, final_df , outlier_removed_stats
    df = New_Refit_routing.copy() 
    neglect = [target] + [cols for cols in df.columns if df[cols].nunique() <= 2] #remove binary and target variable
    cols = df.columns.difference(neglect)  # Getting all columns except the ones in []

    #REMOVE OUTIERS#
    df[cols] = df[cols].apply(lambda col: col.clip(lower = col.quantile(.01), 
                                        upper = closest(col[col < col.quantile(.99)].dropna().values, 
                                        col.quantile(.99))),axis = 0)

    outlier_removed_stats = df.describe().T
    remove_feature = list(outlier_removed_stats[(outlier_removed_stats["mean"]==outlier_removed_stats["max"]) & 
                        (outlier_removed_stats["mean"]==outlier_removed_stats["min"])].index)
    outlier_removed_stats = pd.concat([outlier_removed_stats, df.quantile(q = [.01, .05, .95, .99]).T.rename(columns = {0.01: '1%', 0.05: '5%', 0.95: '95%', 0.99:'99%'})], axis = 1)
    outlier_removed_stats = outlier_removed_stats[['count', 'mean', 'std', 'min', '1%', '5%' ,'25%', '50%', '75%', '95%', '99%', 'max']]                    
    outlier_removed_stats = outlier_removed_stats.round(4).astype(str)

    neglect += remove_feature
    cols = df.columns.difference(neglect)  # Getting all columns except the ones in []

    
    df[cols] = df[cols].apply(lambda col: pd.cut(col.fillna(np.nan),
                                                bins = pd.interval_range(start=float(np.apply_along_axis(control_min , 0,col.dropna())), end = float(np.apply_along_axis(control_max , 0,col.dropna())), 
                                                periods = 10), include_lowest=True).cat.add_categories(pd.Categorical(f"Missing_{col.name}")).fillna(f"Missing_{col.name}"), axis=0)


    test = pd.concat([df[cols].value_counts(normalize = True) for cols in df[cols]], axis = 1)
    cols = test.columns
    test = test.reset_index().melt(id_vars="index", 
                                var_name='column', 
                                value_name='value').dropna().reset_index(drop = True)


    test = test.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Count_Pct"})
    test.Count_Pct = test.Count_Pct.round(4)
    test.IntervalCuts = test.IntervalCuts.astype(str)
    test.IntervalCuts = test.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(").strip("-")[0]).isdigit() else x)

    test2 = pd.concat([df.groupby(col)[target].mean().fillna(0) for col in df[cols]], axis = 1)
    test2.columns = cols
    test2 = test2.reset_index().melt(id_vars="index", var_name='column', value_name='value').dropna().reset_index(drop = True)
    test2 = test2.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Bad_Rate_Pct"})
    test2.Bad_Rate_Pct = test2.Bad_Rate_Pct.round(4)
    test2.IntervalCuts = test2.IntervalCuts.astype(str)
    test2.IntervalCuts = test2.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(").strip("-")[0]).isdigit() else x)


    test["index"] = test["feature"] + "_" + test["IntervalCuts"]
    test = test.set_index("index").sort_index()
    test2["index"] = test2["feature"] + "_" + test2["IntervalCuts"]
    test2 = test2.set_index("index").sort_index()
    final_df = pd.merge(test2, test[test.columns.difference(test2.columns)], on = "index")
   

## QCUT ##
def qcuts_(target):
    global test_q, test2_q, final_df_q
    df2 = New_Refit_routing.copy()
    neglect = [target] + [cols for cols in df2.columns if df2[cols].nunique() <= 2] #remove binary and target variable
    cols = df2.columns.difference(neglect)  # Getting all columns except the ones in []

    #DEBUGGING CODE#####################################################################################
    # for i in df2[cols].columns:
    #     print(i)
    #     print(df2[i][df2[i] < df2[i].quantile(.99)].dropna().values)
    #     print(df2[i].quantile(.99))
    #     print(closest(df2[i][df2[i] < df2[i].quantile(.99)].dropna().values, df2[i].quantile(.99)))
        # df2.apply(lambda col: col.clip(lower = col.quantile(.01), 
        #                                 upper = closest(col[col < col.quantile(.99)].dropna().values, 
        #                                 col.quantile(.99))),axis = 0)
    
    ####################################################################################################
    #REMOVE OUTIERS#

    df2[cols] = df2[cols].apply(lambda col: col.clip(lower = col.quantile(.01), 
                                        upper = closest(col[col < col.quantile(.99)].dropna().values, 
                                        col.quantile(.99))),axis = 0)

    temp = df2.describe().T
    remove_feature = list(temp[(temp["mean"]==temp["max"]) & 
                        (temp["mean"]==temp["min"])].index)

    neglect+= remove_feature
    cols = df2.columns.difference(neglect)  # Getting all columns except the ones in []
    # rank(method='first') is a must in qcut 
    # df2[cols] = df2[cols].apply(lambda col: pd.qcut(col.fillna(np.nan).rank(method='first'),
    #                                                 q = 10, duplicates = "drop").cat.add_categories(pd.Categorical(f"Qcut_Missing_{col.name}")).fillna(f"Qcut_Missing_{col.name}"), axis=0)
    df2[cols] = df2[cols].apply(lambda col: pd.qcut(col.fillna(np.nan).rank(method='first'),q = 10, labels=range(1,11)).cat.rename_categories({10:"Last"}).astype(str).replace(dict(dict(pd.concat([col,
           pd.qcut(col.fillna(np.nan).rank(method='first'),q = 10, labels=range(1,11)).cat.rename_categories({10:"Last"})
           .apply(str)], axis = 1, keys= ["feature", "qcuts"]).groupby("qcuts").agg([min, max]).reset_index().astype(str).set_index("qcuts",drop = False)
     .apply(lambda x :x[0]+"_"+"("+str(round(float(x[1]),2))+","+str(round(float(x[2]),2))+"]",axis = 1)),**{"nan":f"Qcut_Missing_{col.name}"})), axis=0)

    test_q = pd.concat([df2[cols].value_counts(normalize = True) for cols in df2[cols]], axis = 1)
    cols = test_q.columns
    test_q = test_q.reset_index().melt(id_vars="index", 
                                var_name='column', 
                                value_name='value').dropna().reset_index(drop = True)


    test_q = test_q.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Count_Pct"})
    test_q.Count_Pct = test_q.Count_Pct.round(4)
    test_q.IntervalCuts = test_q.IntervalCuts.astype(str)
    # test_q.IntervalCuts = test_q.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(")[0]).isdigit() else x)


    test2_q = pd.concat([df2.groupby(col)[target].mean().fillna(0) for col in df2[cols]], axis = 1)
    test2_q.columns = cols
    test2_q = test2_q.reset_index().melt(id_vars="index", var_name='column', value_name='value').dropna().reset_index(drop = True)
    test2_q = test2_q.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Bad_Rate_Pct"})
    test2_q.Bad_Rate_Pct = test2_q.Bad_Rate_Pct.round(4)
    test2_q.IntervalCuts = test2_q.IntervalCuts.astype(str)
    # test2_q.IntervalCuts = test2_q.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(")[0]).isdigit() else x)

    test_q["index"] = test_q["feature"] + "_" + test_q["IntervalCuts"]
    test_q = test_q.set_index("index").sort_index()
    test2_q["index"] = test2_q["feature"] + "_" + test2_q["IntervalCuts"]
    test2_q = test2_q.set_index("index").sort_index()
    final_df_q = pd.merge(test2_q, test_q[test_q.columns.difference(test2_q.columns)], on = "index")
    



@pn.depends(button.param.clicks)
def run(_):
    target, New_Refit_routing = get_data()
    if target == None:
        target = str(selector.value[0])
    else:
        target = "default"
    print(str(selector.value[0]))
    print(target)
    # print(type(file_input.value))
    # print(type(New_Refit_routing))
    print(New_Refit_routing.head())

    stats_()
    cuts_(target)
    qcuts_(target)
    test2_plot = test2.set_index("IntervalCuts").hvplot.scatter(yaxis = "left", y = "Bad_Rate_Pct",
            groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
            width = 1000, title = "Features Segments Cuts by Count", legend = True,label = "Bad Rate(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, 
                                                                                        framewise=True, color = "red", legend_position='top_right')
    test_plot = test.set_index("IntervalCuts").hvplot.bar(y = "Count_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Cuts by Count", legend=True, alpha=0.3, label ="Equal Intervals Data Points(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, framewise=True, yaxis='left')
    final_table = final_df.hvplot.table(groupby = "feature", width=400)

    test2_plot_q = test2_q.set_index("IntervalCuts").hvplot.scatter(yaxis = "left", y = "Bad_Rate_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Q_Cuts by Count", legend = True).opts(xrotation=45, yformatter = "%.04f",show_grid=True, 
                                                                                                framewise=True, color = "red")
    test_plot_q = test_q.set_index("IntervalCuts").hvplot.bar(y = "Count_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Q_Cuts by Count", legend=True, alpha=0.3, label ="Equal Population Data Points(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, framewise=True, yaxis='left')
    final_table_q = final_df_q.hvplot.table(groupby = "feature", width=400)


    stats_table = stats.reset_index().hvplot.table(width = 1000,title="Summary Statistics of the Data", hover = True, responsive=True, 
                    shared_axes= False, fit_columns = True,
                    padding=True, height=500, index_position = 0, fontscale = 1.5)
    stats_table_no_outliers = outlier_removed_stats.reset_index().hvplot.table(width = 1000,title="Summary Statistics of the Capped Outliers Data", hover = True, responsive=True, 
                    shared_axes= False, fit_columns = True,
                    padding=True, height=500, index_position = 0, fontscale = 1.5)
    #PANEL
    pn.extension( template="fast")
    pn.state.template.param.update(
        # site_url="",
        site="CreditRisk",
        title="Feature Distribution & Statistics",
        # favicon="https://raw.githubusercontent.com/opploans/DS_modelling_tools/main/docs/Resources/favicon.ico?token=GHSAT0AAAAAABYR5F6VDZ2PU33UY6NN7NQEY3C2ASA"
        # favicon="",
    )
    
    title = pn.pane.Markdown(
    """
    ### Feature Distribution (Bin Count & Bad Rate)
    """,
    width=800,
    )

    return pn.Column(
                title,
                (test2_plot * test_plot * test2_plot_q * test_plot_q + (final_table + final_table_q)).cols(3),
                (stats_table + stats_table_no_outliers).cols(2),
            )



profiles = '''
### Other Web Apps:

* [Twitter Sentiment Analysis Flask App](https://firobeid.pythonanywhere.com/)

* [Personal Lectures @ UCBerkley Using Panel App](https://firobeid.github.io/compose-plots/script.html)
'''
pn.Row(pn.Column(widgets, profiles), pn.layout.Spacer(width=20), run).servable(target='main')