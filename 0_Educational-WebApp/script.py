import numpy as np
import panel as pn
from pathlib import Path
import pandas as pd
import hvplot.pandas
from io import BytesIO, StringIO
import sys
import time
from dotenv import load_dotenv
import os
load_dotenv()
'''
<meta http-equiv="pragma" content="no-cache" />
<meta http-equiv="expires" content="-1" />
'''

hospital_data = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/hospital_claims.csv'
).dropna()

# Slice the DataFrame to consist of only "552 - MEDICAL BACK PROBLEMS W/O MCC" information
procedure_552_charges = hospital_data[
    hospital_data["DRG Definition"] == "552 - MEDICAL BACK PROBLEMS W/O MCC"
]
# Group data by state and average total payments, and then sum the values
payments_by_state = procedure_552_charges[["Average Total Payments", "Provider State"]]
# Sum the average total payments by state
total_payments_by_state = payments_by_state.groupby("Provider State").sum()
plot1 = total_payments_by_state.hvplot.bar(rot = 45)


# Sort the state data values by Average Total Paymnts
sorted_total_payments_by_state = total_payments_by_state.sort_values("Average Total Payments")
sorted_total_payments_by_state.index.names = ['Provider State Sorted']
# Plot the sorted data
plot2 = sorted_total_payments_by_state.hvplot.line(rot = 45)

sorted_total_payments_by_state.index.names = ['Provider State Sorted']
plot3 = total_payments_by_state.hvplot.bar(rot = 45) + sorted_total_payments_by_state.hvplot(rot = 45)

# Group data by state and average medicare payments, and then sum the values
medicare_payment_by_state = procedure_552_charges[["Average Medicare Payments", "Provider State"]]
total_medicare_by_state = medicare_payment_by_state.groupby("Provider State").sum()
# Sort data values
sorted_total_medicare_by_state = total_medicare_by_state.sort_values("Average Medicare Payments")
plot4 = sorted_total_medicare_by_state.hvplot.bar(rot = 45)

plot5 = sorted_total_payments_by_state.hvplot.line(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", rot = 45)

# Overlay plots of the same type using * operator
plot6 = sorted_total_payments_by_state.hvplot.bar(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", width = 1000, rot = 45)

# hvplot_snip = pn.pane.HTML("https://firobeid.github.io/compose-plots/Resources/binning_V1.html")
hvplot_snip = pn.pane.Markdown("""[DataViz HTMLS Deployments](https://firobeid.github.io/compose-plots/Resources/binning_V1.html)""")
pn.extension( template="fast")

pn.state.template.param.update(
    # site_url="",
    # site="",
    title="UCBerkely FinTech Bootcamp Demo",
    favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
# Create a Title for the Dashboard
title = pn.pane.Markdown(
    """
# UCBerkley FinTech Bootcamp Demo - Firas Obeid
""",
    width=1000,
)

title_0 = pn.pane.Markdown(
    """
# Intro to Python : Text Munging & Cleaning
""",
    width=800,  
)

title1 = pn.pane.Markdown(
    """
# Hospital Data Analysis
""",
    width=800,  
)

title2 = pn.pane.Markdown(
    """
# Machine Learning Unwinding 
""",
    width=800,  
)

image = pn.pane.image.PNG(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    alt_text='Meme Logo',
    link_url='https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    width=500
)
welcome = pn.pane.Markdown(
    """
### This dashboard/WebApp leverages FinTech and Data Science tools for practical and hands on demo's for UCBerkley FinTech Bootcamp students in [`Firas Ali Obeid's`](https://www.linkedin.com/in/feras-obeid/) classes
* Motive is to keep students up to date with the tools that allow them to define a problem till deployment in a very short amount of time for efficient deliverables in the work place or in academia. 
* The tool/web app is developed completly using python and deployed serverless on github pages (not static anymore right?! 

* Disclaimer: All data presented are from UCBerkley resources.
* Disclaimer: All references: https://blog.holoviz.org/panel_0.14.html

***`Practice what you preach`***

"""
)

##Python Competition##
python_intro = pn.pane.Markdown(
    """
# Essence of Data Cleaning in Python

* Write a script/function that ingests the following list and addresses all unique string formatting:
```
names = ['St. Albans',
        'St. Albans', 
        'St Albans', 
        'St.Ablans',
        "St.albans", 
        "St. Alans", 'S.Albans',
        'St..Albans', 'S.Albnas', 
        'St. Albnas', "St.Al bans", 'St.Algans',
        "Sl.Albans", 'St. Allbans', "St, Albans", 'St. Alban', 'St. Alban']
```

* The intended output is the following, where you clean and split all into `Sx Axxxx` in one shot:
```
['St Albans', 'St Albans', 'StAlbans', 'St Ablans', 
 'St Albans', 'St Alans', 'S Albans', 'St Albans', 'S Albnas', 
 'St Albnas', 'St Albans', 'St Algans', 'Sl Albans', 'St Allbans', 'St Albans', 'St Alban', 'St Alban']
```


***`Cleaning text without using any package`***

"""
)
code_submission = pn.widgets.TextAreaInput(value="", height=300, name='Paste your code below (remember to print(results)')
run_python_comp = pn.widgets.Button(name="Click to Check Code Runtime/Accuracy Results")

def time_it():
    return  pd.to_datetime(time.time(),unit = 's')
# def memory()->str:
#     psutil
#     return print('used: {}% free: {:.2f}GB'.format(psutil.virtual_memory().percent, float(psutil.virtual_memory().free)/1024**3))#@ 

def python_competition():
    names = ['St. Albans',
        'St. Albans', 
        'St Albans', 
        'St.Ablans',
        "St.albans", 
        "St. Alans", 'S.Albans',
        'St..Albans', 'S.Albnas', 
        'St. Albnas', "St.Al bans", 'St.Algans',
        "Sl.Albans", 'St. Allbans', "St, Albans", 'St. Alban', 'St. Alban']
    actual_output = ['St Albans', 'St Albans', 'St Albans', 'St Ablans','St Albans', 'St Alans', 'S Albans', 'St Albans', 'S Albnas', 
                     'St Albnas', 'St Albans', 'St Algans', 'Sl Albans', 'St Allbans', 'St Albans', 'St Alban', 'St Alban']

    if str(code_submission.value) == "":
        # return pn.pane.Markdown(f"""""")
        return pn.pane.Alert("""###Please pass in your code above!""", alert_type="warning",)
    try:
        # code = str(code_submission.value.decode("utf-8"))
        code = code_submission.value
        # print(code)
        # create file-like string to capture output
        codeOut = StringIO()
        codeErr = StringIO()
        # capture output and errors
        sys.stdout = codeOut
        sys.stderr = codeErr
        start = time_it()
        # start_memory = float(psutil.virtual_memory().free)/1024**3
        exec(code)
        end = time_it()
        # end_memory = float(psutil.virtual_memory().free)/1024**3
        loop_time = end - start
        # loop_memory = start_memory - end_memory
        # restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # s = codeErr.getvalue()
        # print("error:\n%s\n" % s)
        s = codeOut.getvalue()
        s = eval(s)

        codeOut.close()
        codeErr.close()
        accuracy = len(set(s).intersection(set(actual_output)))/len(set(actual_output))
        results = pd.DataFrame({'Results(Time+Space_Complexity':{'Nanoseconds': loop_time.nanoseconds, 'Microseconds': loop_time.microseconds
                            ,'Seconds': loop_time.seconds,'Total_Seconds':loop_time.total_seconds() ,'Accuracy': '%.2f' % (accuracy*100)}}) #'Memory': '%d MB' % (loop_memory* 1024),
        return pn.widgets.DataFrame(results.sort_index(), width=600, height=1000, name = 'Results')
    except Exception as e: 
        return pn.pane.Markdown(f"""{e}""")

py_widgets_submission = pn.WidgetBox(
    pn.panel("""# Check your Code""", margin=(0, 10)),
    pn.panel('* Past your code below, no need to to add the original list.', margin=(0, 10)),
    pn.panel('* Please end your code with a print() of your results list. Only put one print() at the end and no other print() should exist', margin=(0, 10)),
    pn.panel('* If you got an error, remove all spaces between consecutive lines', margin=(0, 10)),
    code_submission,
    run_python_comp, 
    pn.pane.Alert("""##                Your Code Submission Results""", alert_type="success",),
    width = 500
)


@pn.depends(run_python_comp.param.clicks)
def python_competition_submission(_):
    return pn.Column(python_competition)

#ML GENERAL
ml_slider = pn.widgets.IntSlider(start=1, end=10)
def ml_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/{index}.png"
    return pn.pane.JPG(url, width = 500)

ml_output = pn.bind(ml_slideshow, ml_slider)
ml_app = pn.Column(ml_slider, ml_output)

##DATA
data_slider = pn.widgets.IntSlider(start=1, end=8)
def data_slideshow(index):
    url2 = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/Data_Splitting/{index}.png"
    return pn.pane.PNG(url2,width = 800)
data_output = pn.bind(data_slideshow, data_slider)

##CLUSTERING
clustering_slider = pn.widgets.IntSlider(start=1, end=36)
def cluster_slideshow(index):
    url2 = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/Clustering/Clustering-{index}.png"
    return pn.pane.PNG(url2,width = 800)
cluster_output = pn.bind(cluster_slideshow, clustering_slider)
# cluster_app = pn.Column(clustering_slider, cluster_output)
k_means_simple = pn.pane.Markdown("""
### K_means Simple Algo Implementation
```python

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels
```
""",width = 500)

##GENERAL ML
general_ml_slider = pn.widgets.IntSlider(start=1, end=40)
def general_ml_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Algo_Survey/{index}.png"
    return pn.pane.PNG(url,width = 800)
general_ml_output = pn.bind(general_ml_slideshow, general_ml_slider)

ML_quote = pn.pane.Markdown(
    """
***`"When your fundraising it's AI
When you're hiring it is ML
When you're implementing it's Linear Regression
When you are debugging it's printf()" - Barron Schwartz`***
"""
)

ML_algoes = pn.pane.Markdown("""
### Some behind the Scenes Simple Implementations
```python

import numpy as np

def LogesticRegression_predict(features, weights, intercept):
    dot_product = np.dot(features,weights.T) #or .reshape(-1) instead of T
    z = intercept + dot_product 
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

import pickle
def save_model(model_name, model):
    '''
    model_name = name.pkl
    joblib.load('name.pkl')
    assign a variable to load model
    '''
    with open(str(model_name), 'wb') as f:
        pickle.dump(model, f)
```

### Criteria for Splitting in Decision Tress
```python
def gini(rows):
    '''
    Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    '''
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
```
### Find Best Split Algo (Decision Tree)

```python
def find_best_split(rows):
    '''Find the best question to ask by iterating over every feature / value
    and calculating the information gain.'''
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question
```

#### Why we doing Label Encoding?
- We apply One-Hot Encoding when:

The categorical feature is not ordinal (like the countries above)
The number of categorical features is less so one-hot encoding can be effectively applied

- We apply Label Encoding when:

The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
The number of categories is quite large as one-hot encoding can lead to high memory consumption
```python
categorical_vars = list(df.columns[df.dtypes == object].values)
obj_df = df.select_dtypes(include=['object']).copy() 
map_dict = {col: {n: cat for n, cat in enumerate(obj_df[col].astype('category').cat.categories)} for col in obj_df}
obj_df = pd.DataFrame({col: obj_df[col].astype('category').cat.codes for col in obj_df}, index=obj_df.index)

```
""",width = 800)

ML_metrics =  pn.pane.Markdown("""
### Binary Classification Metrics Calculation

```python
__author__: Firas Obeid
def metrics(confusion_matrix):
    '''
    Each mean is appropriate for different types of data; for example:

    * If values have the same units: Use the arithmetic mean.
    * If values have differing units: Use the geometric mean.
    * If values are rates: Use the harmonic mean.
    confusion_matrix = [[ TN, FP ],
                        [ FN, TP ]]
    '''
    TN = matrix[0,0]
    FP = matrix[0,1]
    FN = matrix[1,0]
    TP = matrix[1,1]
    Specificity =  round(TN / (FP + TN), 4) # True Negative Rate 
    FPR  = round(FP / (FP + TN), 4)
    Confidence = round(1 - FPR, 4)
    FDR = round(FP / (FP + TP), 4)
    Precision = 1 - FDR # TP / (FP + TP)
    Recall_Power = round(TP / (TP + FN), 4) #Sensitivity or TPR
    G_mean = (Specificity * Recall_Power) **(1/2) 
    Accuracy = round((TP + TN) / (TP +FP + TN + FN), 4)
    return {'FPR':FPR, 'Confidence': Confidence, 'FDR' :FDR, 'Precision': 
            Precision, 'Recall_Power':Recall_Power, 'Accuracy': Accuracy, "G_mean": G_mean}
```
""", width = 800)

knn_scratch =  pn.pane.Markdown("""
### K-Nearest Neighbor from Scratch
```python
__author__: Mohammad Obeid
import numpy as np
def knn(X_train, y_train, X_test, y_test, k):
    '''
    returns the test error compared to the predicted labels
    '''
    distances = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train) ** 2, axis=2))
    y_pred = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        # Get the indices of the k-nearest neighbors
        indices = np.argsort(distances[i])[:k]
        
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = y_train[indices].astype(int)
        
        # Predict the label of test sample i
        y_pred[i] = np.bincount(k_nearest_labels).argmax()
    
    return sum(y_pred != y_test)/len(y_test)
## Optimal K for KNN
test_errors = []
for k in range(1, 101):
    test_error = knn(X_train, y_train, X_test, y_test, k)
    test_errors.append(test_error)

# Plot the test errors as a function of n_neighbors
plt.plot(range(1, 101), test_errors)
plt.xlabel('n_neighbors')
plt.ylabel('Test error')
plt.title('KNN classifier performance')
plt.show()
```
""", width = 880)
prec_recall =  pn.pane.Markdown("""
### Precision VS Recall Interpretation
** Recall : How many of that class (1 or 0) does the model capture?

** Precision: How many are of those captured are correct prediction? 

On a high level, Recall controls False Negative (i.e more important in medical field)
and Precision controls False Positived (i.e more important in credit risk, traind, finance in general...)
In marketing for example, we care about a balance between precision & Recall (We dont want to have high 
recall meaning we would reach out to customers that we predict incorrectly eventually due to the low precision)
""", width = 500)

##DEEP LEARNING
dl_slider = pn.widgets.IntSlider(start=1, end=7)
def dl_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/DL/{index}.png"
    return pn.pane.PNG(url,width = 800)
dl_output = pn.bind(dl_slideshow, dl_slider)
lstm_gif = pn.pane.GIF('https://raw.githubusercontent.com/firobeid/firobeid.github.io/blob/main/docs/compose-plots/Resources/ML_lectures/DL/LSTM_ANIMATION.gif', alt_text = ' LSTM Animation')
DL_tips = pn.pane.Markdown("""
### Binary CrossEntropy Error (Loss/Error)
```python
def Z(x):
    return np.log(x / (1-x))
def error(Z, Y):
    return(max(Z,0) - Z*Y + np.log(1 + np.exp(-abs(Z))))

y_pred = nn.predict(X_test)
y_pred = np.array(list(map(Z, y_pred)))
error = np.vectorize(error)

all_errors = error(y_pred.ravel(), y_test)

np.mean(all_errors)
```

# What I Learned from my Personal Research?

* If faced with [Failed to call ThenRnnBackward]:

1. Allowing GPU Memory Growth
2. Using batch_input_shape instead of input_shape
3. Using drop_remainder=True when creating batches

* If faced with Crashing IPyhton during Training:

1. Simply put verbose=0 in all model.fit(...) instructions
2. Install keras-tqdm to manage progress bar
3. Redirect the output to a file

### Modelling Tips for Neural Netwroks
We can specify devices for storage and calculation, such as the CPU or GPU. By default, data are created in the main memory and then use the CPU for calculations.

The deep learning framework requires all input data for calculation to be on the same device, be it CPU or the same GPU.

You can lose significant performance by moving data without care. A typical mistake is as follows: computing the loss for every minibatch on the GPU and reporting it back to the user on the command line (or logging it in a NumPy ndarray) will trigger a global interpreter lock which stalls all GPUs. It is much better to allocate memory for logging inside the GPU and only move larger logs.

- For Tensorflow-2: You can just use LSTM with no activation specified (ied default to tanh) function and it will automatically use the CuDNN version
- Gradient clipping is a technique to prevent exploding gradients in very deep networks, usually in recurrent neural networks. ... This prevents any gradient to have norm greater than the threshold and thus the gradients are clipped.

- PCO to intialize weights help in time computation reduction and global optima finding
- Denoising input data helps predict small price changes
- Epoch means one pass over the full training set
- Batch means that you use all your data to compute the gradient during one iteration.
- Mini-batch means you only take a subset of all your data during one iteration.
- In the context of SGD, "Minibatch" means that the gradient is calculated across the entire batch before updating weights. If you are not using a "minibatch", every training example in a "batch" updates the learning algorithm's parameters independently.

- Batch Gradient Descent. Batch size is set to the total number of examples in the training dataset. (batch_size = len(train))
- Stochastic Gradient Descent. Batch size is set to one. (batch_size = 1)
- Minibatch Gradient Descent. Batch size is set to more than one and less than the total number of examples in the training dataset. (batch_size = 32,64...)

### Tips for Activation Functions:
- When using the ReLU function for hidden layers, it is a good practice to use a "He Normal" or "He Uniform" weight initialization and scale input data to the range 0-1 (normalize) prior to training.
- When using the Sigmoid function for hidden layers, it is a good practice to use a "Xavier Normal" or "Xavier Uniform" weight initialization (also referred to Glorot initialization, named for Xavier Glorot) and scale input data to the range 0-1 (e.g. the range of the activation function) prior to training.
- When using the TanH function for hidden layers, it is a good practice to use a "Xavier Normal" or "Xavier Uniform" weight initialization (also referred to Glorot initialization, named for Xavier Glorot) and scale input data to the range -1 to 1 (e.g. the range of the activation function) prior to training.

#### Tips for LSTM Inputs 
- The LSTM input layer must be 3D.
- The meaning of the 3 input dimensions are: samples, time steps, and features (sequences, sequence_length, characters).
- The LSTM input layer is defined by the input_shape argument on the first hidden layer.
- The input_shape argument takes a tuple of two values that define the number of time steps and features.
- The number of samples is assumed to be 1 or more.
- The reshape() function on NumPy arrays can be used to reshape your 1D or 2D data to be 3D.
- The reshape() function takes a tuple as an argument that defines the new shape
- The LSTM return the entire sequence of outputs for each sample (one vector per timestep per sample), if you set return_sequences=True.
- Stateful RNN only makes sense if each input sequence in a batch starts exactly where the corresponding sequence in the previous batch left off. Our RNN model is stateless since each sample is different from the other and they dont form a text corpus but are separate headlines.

#### Tips for Embedding Layer
- Gives relationship between characters.
- Dense vector representation (n-Dimensional) of float point values. Map(char/byte) to a dense vector.
- Embeddings are trainable weights/paramaeters by the model equivalent to weights learned by dense layer.
- In our case each unique character/byte is represented with an N-Dimensional vector of floating point values, where the learned embedding forms a lookup table by "looking up" each characters dense vector in the table to encode it.
- A simple integer encoding of our characters is not efficient for the model to interpret since a linear classifier only learns the weights for a single feature but not the relationship (probability distribution) between each feature(characters) or there encodings.
- A higher dimensional embedding can capture fine-grained relationships between characters, but takes more data to learn.(256-Dimensions our case)


""", width = 1000)
##TIMESERIES
timeseries_libs = pn.pane.Markdown("""
## 10 Time-Series Python Libraries in 2022:

### 📚 Flow forecast

Flow forecast is a deep learning for time series forecasting framework. It provides the latest models (transformers, attention models, GRUs) and cutting edge concepts with interpretability metrics. It is the only true end-to-end deep learning for time series forecasting framework.

### 📚 Auto_TS

Auto_TS train multiple time series models with just one line of code and is a part of autoML.

### 📚 SKTIME

Sktime an extension to scikit-learn includes machine learning time-series for regression, prediction, and classification. This library has the most features with interfaces scikit-learn, statsmodels, TSFresh and PyOD.

### 📚 Darts

Darts contains a large number of models ranging from ARIMA to deep neural networks. It also lets users combine predictions from several models and external regressors which makes it easier to backtest models.

### 📚 Pmdarima

Pmdarima is a wrapper over ARIMA with automatic Hyperparameter tunning for analyzing, forecasting, and visualizing time series data including transformers and featurizers, including Box-Cox and Fourier transformations and a seasonal decomposition tool.

### 📚 TSFresh

TSFresh automates feature extraction and selection from time series. It has Dimensionality reduction, Outlier detection and missing values.

### 📚 Pyflux

Pyflux builds probabilistic model, very advantageous for tasks where a more complete picture of uncertainty is needed and the latent variables are treated as random variables through a joint probability.


### 📚 Prophet

Facebook's Prophet is a forecasting tool for CSV format and is suitable for strong seasonal data and robust to missing data and outliers.
Prophet is a library that makes it easy for you to fit a model that decomposes a time series model into trend, season, and holiday components. It's somewhat customizable and has a few nifty tools like graphing and well-thought out forecasting.
Prophet does the following linear decomposition:

* g(t): Logistic or linear growth trend with optional linear splines (linear in the exponent for the logistic growth). The library calls the knots 'change points.'
* s(t): Sine and cosine (i.e. Fourier series) for seasonal terms.
* h(t): Gaussian functions (bell curves) for holiday effects (instead of dummies, to make the effect smoother).

[Some thoughts about Prophet](https://www.reddit.com/r/MachineLearning/comments/syx41w/p_beware_of_false_fbprophets_introducing_the/)

### 📚 Statsforecast
[GitHub Link to Statsforecast](https://github.com/Nixtla/statsforecast)

Statsforecast offers a collection of univariate time series. It invludes ADIDA, HistoricAverage, CrostonClassic, CrostonSBA, CrostonOptimized, SeasonalNaive, IMAPA Naive, RandomWalkWithDrift, TSB, AutoARIMA and ETS.
Impressive fact: It is 20x faster than pmdarima , 500x faster than Prophet,100x faster than NeuralProphet, 4x faster than statsmodels. 

### 📚 PyCaret

PyCaret replaces hundreds of lines of code with few lines only. Its time-series forecasting is in pre-release mode with --pre tag with 30+ algorithms. It includes automated hyperparameter tuning, experiment logging and deployment on cloud.

### 📚 NeuralProphet

NeuralProphet is a Neural Network based Time-Series model, inspired by Facebook Prophet and AR-Net, built on PyTorch.

Source: Maryam Miradi, PhD 
""",width = 800)
timeseries_data_split = pn.pane.Markdown("""
### Training and Validating Time Series Forecasting Models
```python

from sklearn.model_selection import TimeSeriesSplit
N_SPLITS = 4


X = df['timestamp']
y = df['value']


folds = TimeSeriesSplit(n_splits = N_SPLITS)


for i, (train_index, valid_index) in enumerate(folds.split(X)):
	X_train, X_valid = X[train_index], X[valid_index]
	y_train, y_valid = y[train_index], y[valid_index]
```
### Training and Validating `Financial` Time Series Forecasting Models
```python

__author__ = 'Stefan Jansen'
class MultipleTimeSeriesCV:
    '''
    Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes
    '''

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
```
""",width = 800)
ts_gif = pn.pane.GIF("https://raw.githubusercontent.com/firobeid/machine-learning-for-trading/main/assets/timeseries_windowing.gif")
ts_cv = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ts/cv.png",link_url = 'https://wandb.ai/iamleonie/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting/reports/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting--VmlldzoyNjkxOTMz', width = 800)
# Create a tab layout for the dashboard
# https://USERNAME.github.io/REPO_NAME/PATH_TO_FILE.pdf
motivational = pn.pane.Alert("## YOUR PROGRESS...\nUpward sloping and incremental. Keep moving forward!", alert_type="success")
gif_pane = pn.pane.GIF('https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif')
progress_ = pn.pane.PNG('https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/Progress.png')

##########################
##TIMESERIES COMPETITION##
##########################
reward = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/TimeSeriesCompetition/Images/Reward.png")
other_metrics = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ts/Regression_Loss_functions.png", height = 500)
def cal_error_metrics():
    global real_test_data, predictions, rmse_error

    def rmse(preds,target):
        if (len(preds)!=len(target)):
            raise AttributeError('list1 and list2 must be of same length')
        return round(((sum((preds[i]-target[i])**2 for i in range(len(preds)))/len(preds)) ** 0.5),2)

    try:
        assert len(real_test_data) == len(predictions)
    except Exception as e: # if less than 2 words, return empty result
        return pn.pane.Markdown("""ERROR:You didnt upload excatly 17519 predictions rows!!""")
    try:
        rmse_error = rmse(real_test_data["GHI"].values, predictions[predictions.columns[0]].values)

        error_df = pd.DataFrame({"RMSE":[rmse_error]}, index = [str(file_input_ts.filename)])
        error_df.index.name = 'Uploaded_Predictions'
    except Exception as e: 
        return pn.pane.Markdown(f"""{e}""")

    return pn.widgets.DataFrame(error_df, width=300, height=100, name = 'Score Board')


def get_real_test_timeseries():
    global real_test_data, predictions 
    real_test_data = hospital_data = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/TimeSeriesCompetition/test_data/competition_real_test_data_2018.csv'
).dropna()
    if file_input_ts.value is None:
        predictions = pd.DataFrame({'GHI': [real_test_data['GHI'].mean()] * len(real_test_data)})
    else:
        predictions = BytesIO()
        predictions.write(file_input_ts.value)
        predictions.seek(0)
        print(file_input_ts.filename)
        try:
            predictions = pd.read_csv(predictions, error_bad_lines=False).dropna()#.set_index("id")
        except:
            predictions = pd.read_csv(predictions, error_bad_lines=False).dropna()
        if len(predictions.columns) > 1:
            predictions = predictions[[predictions.columns[-1]]]
        predictions = predictions._get_numeric_data()
        predictions[predictions < 0] = 0 #predictions cant be hegative for solar energy prediction task
        # New_Refit_routing = New_Refit_routing[[cols for cols in New_Refit_routing.columns if New_Refit_routing[cols].nunique() >= 2]] #remove columns with less then 2 unique values
    # return predictions

def github_cred():
    from github import Github
    repo_name = 'firobeid/TimeSeriesCompetitionTracker'
    # using an access token
    g = Github(os.getenv('GITHUB_TOKEN'))
    return g.get_repo(repo_name)

def leaderboard_ts():
    global file_on_github
    # repo_name = 'firobeid/TimeSeriesCompetitionTracker'
    # # using an access token
    # g = Github(os.getenv('GITHUB_TOKEN'))
    # # Create Github linkage Instance
    # g = github_cred()
    # if prediction_submission_name.value == 'Firas_Prediction_v1':
    repo = github_cred()
    contents = repo.get_contents("")
    competitior_rank_file = 'leadership_board_ts.csv'
    if competitior_rank_file not in [i.path for i in contents]:
        print("Creatine leaderboard file...")
        repo.create_file(competitior_rank_file, "creating timeseries leaderboard", "Competitor_Submission, RMSE", branch="main")
    file_on_github = pd.read_csv("https://raw.githubusercontent.com/firobeid/TimeSeriesCompetitionTracker/main/leadership_board_ts.csv", delim_whitespace=" ") 

def upload_scores():
    global rmse_error, sub_name, file_on_github
    competitior_rank_file = 'leadership_board_ts.csv'
    repo = github_cred()
    submission = sub_name
    score = rmse_error
    leaderboard_ts()
    file_on_github.loc[len(file_on_github.index)] = [submission, score]

    target_content = repo.get_contents(competitior_rank_file)
    repo.update_file(competitior_rank_file, "Uploading scores for %s"%sub_name,  file_on_github.to_string(index=False), target_content.sha, branch="main")
    return pn.pane.Markdown("""Successfully Uploaded to Leaderboard!""")

def final_github():
    global sub_name
    global real_test_data, predictions, rmse_error
    sub_name = str(prediction_submission_name.value.replace("\n", "").replace(" ", ""))
    print(sub_name)
    if 'rmse_error' not in globals(): #not to allow saving rmse everytime site is reoaded
        return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Competitor_Submission'), width=600, height=1000, name = 'Leader Board')
    
    else:
        try:
            if sub_name != 'Firas_Prediction_v1': #not to allow saving rmse everytime site is reoaded also
                upload_scores()
        except Exception as e: 
            return pn.pane.Markdown(f"""{e}""")
        file_on_github["Rank"] = file_on_github.rank(method = "min")["RMSE"]
        return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Rank'), width=600, height=1000, name = 'Leader Board')

run_github_upload = pn.widgets.Button(name="Click to Upload Results to Leaderscore Board!")
prediction_submission_name  = pn.widgets.TextAreaInput(value="Firas_Prediction_v1", height=100, name='Change the name of submission below:')
widgets_submission = pn.WidgetBox(
    pn.panel("""# Submit to LeaderBoard Ranking""", margin=(0, 10)),
    pn.panel('* Change Submision Name Below to your own version and team name (no spaces in between)', margin=(0, 10)),
    prediction_submission_name,
    run_github_upload, 
    pn.pane.Alert("""##                Leader Ranking Board""", alert_type="success",),
    width = 500
)

@pn.depends(run_github_upload.param.clicks)
def ts_competition_submission(_):
    leaderboard_ts()
    return pn.Column(final_github)


run_button = pn.widgets.Button(name="Click to get model scores!")
file_input_ts = pn.widgets.FileInput(align='center')
text_ts = """
# Prediction Error Scoring

This section is to host a time series modelling competition between UCBekely students teams'. The teams should
build a time series univariate or multivariate model but the aim is to forcast the `GHI` column (a solar energy storage metric).

The train data is 30 minutes frequecy data between 2010-2017 for solar energy for UTDallas area. The students then predict the whole off 2018
,which is 17519 data points (periods) into the future (2018). The students submit there predictions as csv over here, 
get error score (RMSE not the best maybe but serves learning objective) and submit to leaderboard to be ranked. Public submissions
are welcome! But I cant give you extra points on project 2 ;)

The data used for the modelling can be found here: 
[Competition Data](https://github.com/firobeid/Forecasting-techniques/tree/master/train_data)

### Instructions
1. Upload predictions CSV (only numerical data)
2. Make sure you have 17519 predictions / row in your CSV and only one column
3. Press `Click to get model error/score!`
4. Observe you predictions error under yellow box bellow
5. If satisfied move on to the next box to the right to submit team name and prediction. 
`My code takes care of pulling your error and storing it on GitHub to be ranked against incoming scores from teams`
"""
widgets_ts = pn.WidgetBox(
    pn.panel(text_ts, margin=(0, 10)),
    pn.panel('Upload Prediction CSV', margin=(0, 10)),
    file_input_ts,
    run_button, 
    pn.pane.Alert("### Prediction Results Will Refresh Below After Clicking above", alert_type="warning")
    , width = 500
)

def update_target(event):
    get_real_test_timeseries()

file_input_ts.param.watch(update_target, 'value')

@pn.depends(run_button.param.clicks)
def ts_competition(_):
    get_real_test_timeseries()
    return pn.Column(cal_error_metrics)
##########################
##  ML COMPETITION      ##
##########################
reward_ml = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/TimeSeriesCompetition/Images/Reward.png")
# other_metrics = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ts/Regression_Loss_functions.png", height = 500)
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 10, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.15 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 10, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def ks(y_real, y_proba):
    from scipy.stats import ks_2samp
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba
    
    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]
    
    ks_ = ks_2samp(class0['proba'], class1['proba'])
    
    
    return ks_[0]

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
  return ece

def save_csv(type_, name):
    from io import StringIO
    sio = StringIO()
    if type_ == 'dev':
        df = pd.read_csv('https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/train_data/dev_data.csv')
    elif type_ == 'test':
        df = pd.read_csv('https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/test_data/test_data.csv')
    df.to_csv(sio)
    sio.seek(0)
    return pn.widgets.FileDownload(sio, embed=True, filename='%s.csv'%name)

def cal_error_metrics_ml():
    global real_test_data_ml, predictions_ml, metrics

    def all_metrics(y_true, y_test):
        from sklearn.metrics import roc_auc_score
        return {"Amex_Metric": amex_metric_mod(y_true,y_test), 
                "KS" : ks(y_true,y_test), 
                "Expected Calibration Error": expected_calibration_error(y_true,y_test), 
                "AUC": roc_auc_score(y_true, y_test)}

    try:
        assert len(real_test_data_ml) == len(predictions_ml)
    except Exception as e: # if less than 2 words, return empty result
        return pn.pane.Markdown(f"""ERROR:You didnt upload excatly {len(real_test_data_ml)} predictions rows!!""")
    try:
        metrics = all_metrics(real_test_data_ml["loan_status"].values, predictions_ml[predictions_ml.columns[0]].values)

        # error_df = pd.DataFrame({"RMSE":[rmse_error]}, index = [str(file_input_ts.filename)])
        error_df = pd.DataFrame({"Metrics_Value":metrics}).T
        error_df.index.name = 'Results'
    except Exception as e: 
        return pn.pane.Markdown(f"""{e}""")

    return pn.widgets.DataFrame(error_df, layout='fit_columns', width=700, height=100, name = 'Score Board')


def get_real_test_labels():
    global real_test_data_ml, predictions_ml 
    real_test_data_ml = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/test_data/test_labels.csv'
).dropna()
    if file_input_ml.value is None:
        predictions_ml = pd.DataFrame({'loan_status': np.random.choice([0,1], size = len(real_test_data_ml), p = [0.87,0.13])})
    else:
        predictions_ml = BytesIO()
        predictions_ml.write(file_input_ml.value)
        predictions_ml.seek(0)
        print(file_input_ml.filename)
        try:
            predictions_ml = pd.read_csv(predictions_ml, error_bad_lines=False).dropna()#.set_index("id")
        except:
            predictions_ml = pd.read_csv(predictions_ml, error_bad_lines=False).dropna()
        if len(predictions_ml.columns) > 1:
            predictions_ml = predictions_ml[[predictions_ml.columns[-1]]]
        predictions_ml = predictions_ml._get_numeric_data()
        # predictions[predictions < 0] = 0 #predictions cant be hegative for solar energy prediction task
        # New_Refit_routing = New_Refit_routing[[cols for cols in New_Refit_routing.columns if New_Refit_routing[cols].nunique() >= 2]] #remove columns with less then 2 unique values
    # return predictions

# def github_cred():
#     # from github import Github
#     repo_name = 'firobeid/TimeSeriesCompetitionTracker'
#     # using an access token
#     g = Github(os.getenv('GITHUB_TOKEN'))
#     return g.get_repo(repo_name)

# def leaderboard_ts():
#     global file_on_github
#     # repo_name = 'firobeid/TimeSeriesCompetitionTracker'
#     # # using an access token
#     # g = Github(os.getenv('GITHUB_TOKEN'))
#     # # Create Github linkage Instance
#     # g = github_cred()
#     # if prediction_submission_name.value == 'Firas_Prediction_v1':
#     repo = github_cred()
#     contents = repo.get_contents("")
#     competitior_rank_file = 'leadership_board_ts.csv'
#     if competitior_rank_file not in [i.path for i in contents]:
#         print("Creatine leaderboard file...")
#         repo.create_file(competitior_rank_file, "creating timeseries leaderboard", "Competitor_Submission, RMSE", branch="main")
#     file_on_github = pd.read_csv("https://raw.githubusercontent.com/firobeid/TimeSeriesCompetitionTracker/main/leadership_board_ts.csv", delim_whitespace=" ") 

# def upload_scores():
#     global rmse_error, sub_name, file_on_github
#     competitior_rank_file = 'leadership_board_ts.csv'
#     repo = github_cred()
#     submission = sub_name
#     score = rmse_error
#     leaderboard_ts()
#     file_on_github.loc[len(file_on_github.index)] = [submission, score]

#     target_content = repo.get_contents(competitior_rank_file)
#     repo.update_file(competitior_rank_file, "Uploading scores for %s"%sub_name,  file_on_github.to_string(index=False), target_content.sha, branch="main")
#     return pn.pane.Markdown("""Successfully Uploaded to Leaderboard!""")

# def final_github():
#     global sub_name
#     global real_test_data_ml, predictions_ml, metrics
#     sub_name = str(prediction_submission_name.value.replace("\n", "").replace(" ", ""))
#     print(sub_name)
#     if 'rmse_error' not in globals(): #not to allow saving rmse everytime site is reoaded
#         return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Competitor_Submission'), width=600, height=1000, name = 'Leader Board')
    
#     else:
#         try:
#             if sub_name != 'Firas_Prediction_v1': #not to allow saving rmse everytime site is reoaded also
#                 upload_scores()
#         except Exception as e: 
#             return pn.pane.Markdown(f"""{e}""")
#         file_on_github["Rank"] = file_on_github.rank(method = "min")["RMSE"]
#         return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Rank'), width=600, height=1000, name = 'Leader Board')

# run_github_upload = pn.widgets.Button(name="Click to Upload Results to Leaderscore Board!")
# prediction_submission_name  = pn.widgets.TextAreaInput(value="Firas_Prediction_v1", height=100, name='Change the name of submission below:')
# widgets_submission = pn.WidgetBox(
#     pn.panel("""# Submit to LeaderBoard Ranking""", margin=(0, 10)),
#     pn.panel('* Change Submision Name Below to your own version and team name (no spaces in between)', margin=(0, 10)),
#     prediction_submission_name,
#     # run_github_upload, 
#     pn.pane.Alert("""##                Leader Ranking Board""", alert_type="success",),
#     width = 500
# )
# def update_submission_widget(event):
#     global sub_name
#     prediction_submission_name.value = event.new
#     sub_name = str(prediction_submission_name.value.replace("\n", "").replace(" ", ""))
#     print(sub_name)
# # when prediction_submission_name changes, 
# # run this function to global variable sub_name
# prediction_submission_name.param.watch(update_submission_widget, "value")

# @pn.depends(run_github_upload.param.clicks)
# def ts_competition_submission(_):
#     leaderboard_ts()
#     return pn.Column(final_github)


run_button_ml = pn.widgets.Button(name="Click to get model scores!")
file_input_ml = pn.widgets.FileInput(align='center')
text_ml = """
# Lending Club Prediction Competition

This section is to host an ML classification competition between UCBerkely student teams'. The teams/individuals should
build classification models to predict the `loan_status` column on test data that I set there respective true labels aside.

This modelling competition coins a full ML model building on a lending club dataset. I downsampled the original
development sample from 2million+ rows to 200k+ rows for practicality. The downsampling preserved the target 
variable distribution as it was done to control the following columns:

`["addr_state", "issue_d", "zip_code", "grade", "sub_grade", "term"]`

The test set,which is 20863 rows is to be passed for predictions on the students champion models. The students check there predictions as csv upload 
over here, to get several metric scores (all metrics below if higher are better except for expected_calibration_error 'lower is better).

The competition data used for the modelling can be found here(right click and copy the link): 

### Download the Train and Test Data'

* [Starter Code](https://github.com/firobeid/firobeid.github.io/blob/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/Start_Code/UCBerkeley_LendingClubData.ipynb)
* [Competition Development Data](https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/train_data/dev_data.csv)
* [Test Unlabled Data](https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Competition/test_data/test_data.csv)

To access the data locally, copy either off the last two hyperlinks and paste them as follows:

```
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/...')
```
### Instructions
1. Upload predictions CSV (only numerical data)
2. Make sure you have 20863 predictions / row in your CSV and only one column
3. Press `Click to get model error/score!`
4. Observe you predictions error under yellow box bellow
5. If satisfied send me your predictions.csv privaetly in slack! 

`My code takes care of pulling your error and storing it on GitHub to be ranked against incoming scores from teams`
"""
widgets_ml = pn.WidgetBox(
    pn.panel(text_ml, margin=(0, 20)),
    # pn.Row(save_csv('dev', 'development_data'),save_csv('test', 'test_data'), width = 500),
    pn.panel('Upload Prediction CSV', margin=(0, 10)),
    file_input_ml,
    run_button_ml, 
    pn.pane.Alert("### Prediction Results Will Refresh Below After Clicking above", alert_type="warning")
    , width = 700
)

def update_target_ml(event):
    get_real_test_labels()

file_input_ml.param.watch(update_target_ml, 'value')

@pn.depends(run_button_ml.param.clicks)
def ml_competition(_):
    get_real_test_labels()
    return pn.Column(cal_error_metrics_ml)


#########
##FINAL##
#########
tabs = pn.Tabs(
    ("Welcome", pn.Column(welcome, image)
    ),
    ("Pythonic Text Munging",pn.Tabs(("Title",pn.Column(pn.Row(title_0))),
                                     ("Coding Competition", pn.Row(python_intro,pn.layout.Spacer(width=20), pn.Column(py_widgets_submission, python_competition_submission)))
                                     )
    
    
    ),
    ("DataViz",pn.Tabs(("Title",pn.Column(pn.Row(title1),hvplot_snip)),
                    ("total_payments_by_state", pn.Row(plot1)),
                    ("sorted_total_payments_by_state", pn.Row(plot2)),
                    ("Tab1 + Tab2", pn.Column(plot3,width=960)),
                    ("sorted_total_medicare_by_state", pn.Row(plot4,plot5, plot6, width=2000))
                      )
    ),
    ("Zen of ML", pn.Tabs(("Title",pn.Row(title2,gif_pane, pn.Column(motivational,progress_))),
                          ('Lets Get Things Straight',pn.Column(ml_slider, ml_output)),
                          ('Data Considerations!!',pn.Column(data_slider, data_output)),
                          ('Unsupervised Learning (Clustering)', pn.Row(pn.Column(clustering_slider, cluster_output),k_means_simple)),
                          ("TimeSeries Forecasting",pn.Row(timeseries_libs,pn.Column(ts_gif, ts_cv),timeseries_data_split)),
                          ("General ML Algorithms' Survey", pn.Row(pn.Column(general_ml_slider, general_ml_output),ML_algoes, pn.Column(knn_scratch, ML_metrics, prec_recall))),
                        #   ('TimeSeries Competition Error Metric',pn.Row(pn.Column(widgets_ts, ts_competition, reward), pn.layout.Spacer(width=20), pn.layout.Spacer(width=20), pn.Column(pn.pane.Markdown("### Other Metrics Can Be Used:"),other_metrics))), 
                          ('TimeSeries Competition Error Metric',pn.Row(pn.Column(widgets_ts, ts_competition, reward), pn.layout.Spacer(width=20), pn.Column(widgets_submission, ts_competition_submission), pn.layout.Spacer(width=20), pn.Column(pn.pane.Markdown("### Other Metrics Can Be Used:"),other_metrics))), 
                          ('ML Classification Competition',pn.Row(pn.Column(widgets_ml, ml_competition, reward), pn.layout.Spacer(width=30), pn.layout.Spacer(width=20), pn.Column(pn.pane.Markdown("### Keep this in mind:"),ML_quote))),
                          ('Neural Netwroks Visit',pn.Row(pn.Column(dl_slider, dl_output), DL_tips))
                         )
    )
    )
    

audio = pn.pane.Audio('http://ccrma.stanford.edu/~jos/mp3/pno-cs.mp3', name='Audio')
pn.Column(pn.Row(title), tabs, pn.Row(pn.pane.Alert("Enjoy some background classic", alert_type="success"),audio), ).servable(target='main')
