```python
# Importing packages
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_profiling
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from array import array
```


```python
df1=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2016.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
df2=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2017.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
df3=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2018.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
df4=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2019.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
df5=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2020.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
df6=pd.read_csv("D:\Downloads\qualite-de-lair-mesuree-dans-la-station-chatelet2021.csv", index_col='DATE/HEURE',parse_dates=True, error_bad_lines=False)
```


```python
df=df6.append([df5, df4, df3, df2, df1])
```


```python
#df.columns =['DATE/HEURE','NON','NO2','PM10','CO2','TEMP','HUMI']
# delete a single row by index value 0
#df = df.drop(labels=0, axis=0)
#df.index= pd.to_datetime(df.index, infer_datetime_format = True, utc = True).astype('datetime64[ns]')
df.index.freq='H'
#df.index = pd.to_datetime(df.index).floor("D")
#df.groupby(df["DATE/HEURE"]).mean()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NO</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-31 23:00:00+01:00</th>
      <td>9</td>
      <td>23</td>
      <td>31</td>
      <td>519</td>
      <td>19.1</td>
      <td>62.1</td>
    </tr>
    <tr>
      <th>2021-12-31 22:00:00+01:00</th>
      <td>18</td>
      <td>28</td>
      <td>39</td>
      <td>526</td>
      <td>18.5</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>2021-12-31 21:00:00+01:00</th>
      <td>37</td>
      <td>36</td>
      <td>51</td>
      <td>589</td>
      <td>18.9</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>2021-12-31 20:00:00+01:00</th>
      <td>21</td>
      <td>33</td>
      <td>49</td>
      <td>589</td>
      <td>18.7</td>
      <td>64.7</td>
    </tr>
    <tr>
      <th>2021-12-31 19:00:00+01:00</th>
      <td>17</td>
      <td>30</td>
      <td>49</td>
      <td>628</td>
      <td>18.8</td>
      <td>64.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.index = df['DATE/HEURE']
#df.drop('DATE/HEURE',axis=1, inplace=True)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NO</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-31 23:00:00+01:00</th>
      <td>9</td>
      <td>23</td>
      <td>31</td>
      <td>519</td>
      <td>19.1</td>
      <td>62.1</td>
    </tr>
    <tr>
      <th>2021-12-31 22:00:00+01:00</th>
      <td>18</td>
      <td>28</td>
      <td>39</td>
      <td>526</td>
      <td>18.5</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>2021-12-31 21:00:00+01:00</th>
      <td>37</td>
      <td>36</td>
      <td>51</td>
      <td>589</td>
      <td>18.9</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>2021-12-31 20:00:00+01:00</th>
      <td>21</td>
      <td>33</td>
      <td>49</td>
      <td>589</td>
      <td>18.7</td>
      <td>64.7</td>
    </tr>
    <tr>
      <th>2021-12-31 19:00:00+01:00</th>
      <td>17</td>
      <td>30</td>
      <td>49</td>
      <td>628</td>
      <td>18.8</td>
      <td>64.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.index = df.index.astype('datetime64[ns]')
df = df.iloc[::-1]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NO</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01 00:00:00+01:00</th>
      <td>32</td>
      <td>37</td>
      <td>139</td>
      <td>649</td>
      <td>23.1</td>
      <td>36.1</td>
    </tr>
    <tr>
      <th>2016-01-01 01:00:00+01:00</th>
      <td>16</td>
      <td>32</td>
      <td>132</td>
      <td>575</td>
      <td>22.9</td>
      <td>35.6</td>
    </tr>
    <tr>
      <th>2016-01-01 02:00:00+01:00</th>
      <td>13</td>
      <td>25</td>
      <td>128</td>
      <td>651</td>
      <td>23.2</td>
      <td>36.3</td>
    </tr>
    <tr>
      <th>2016-01-01 03:00:00+01:00</th>
      <td>14</td>
      <td>27</td>
      <td>125</td>
      <td>668</td>
      <td>23.4</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>2016-01-01 04:00:00+01:00</th>
      <td>17</td>
      <td>32</td>
      <td>93</td>
      <td>652</td>
      <td>23.0</td>
      <td>36.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
pandas_profiling.ProfileReport(df)
```


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_31252\3358060092.py in <module>
    ----> 1 pandas_profiling.ProfileReport(df)
    

    D:\Anaonda\lib\site-packages\IPython\core\displayhook.py in __call__(self, result)
        260             self.start_displayhook()
        261             self.write_output_prompt()
    --> 262             format_dict, md_dict = self.compute_format_data(result)
        263             self.update_user_ns(result)
        264             self.fill_exec_result(result)
    

    D:\Anaonda\lib\site-packages\IPython\core\displayhook.py in compute_format_data(self, result)
        149 
        150         """
    --> 151         return self.shell.display_formatter.format(result)
        152 
        153     # This can be set to True by the write_output_prompt method in a subclass
    

    D:\Anaonda\lib\site-packages\IPython\core\formatters.py in format(self, obj, include, exclude)
        178             md = None
        179             try:
    --> 180                 data = formatter(obj)
        181             except:
        182                 # FIXME: log the exception
    

    D:\Anaonda\lib\site-packages\decorator.py in fun(*args, **kw)
        230             if not kwsyntax:
        231                 args, kw = fix(args, kw, sig)
    --> 232             return caller(func, *(extras + args), **kw)
        233     fun.__name__ = func.__name__
        234     fun.__doc__ = func.__doc__
    

    D:\Anaonda\lib\site-packages\IPython\core\formatters.py in catch_format_error(method, self, *args, **kwargs)
        222     """show traceback on failed format call"""
        223     try:
    --> 224         r = method(self, *args, **kwargs)
        225     except NotImplementedError:
        226         # don't warn on NotImplementedErrors
    

    D:\Anaonda\lib\site-packages\IPython\core\formatters.py in __call__(self, obj)
        343             method = get_real_method(obj, self.print_method)
        344             if method is not None:
    --> 345                 return method()
        346             return None
        347         else:
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in _repr_html_(self)
        436     def _repr_html_(self) -> None:
        437         """The ipython notebook widgets user interface gets called by the jupyter notebook."""
    --> 438         self.to_notebook_iframe()
        439 
        440     def __repr__(self) -> str:
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in to_notebook_iframe(self)
        416         with warnings.catch_warnings():
        417             warnings.simplefilter("ignore")
    --> 418             display(get_notebook_iframe(self.config, self))
        419 
        420     def to_widgets(self) -> None:
    

    D:\Anaonda\lib\site-packages\pandas_profiling\report\presentation\flavours\widget\notebook.py in get_notebook_iframe(config, profile)
         73         output = get_notebook_iframe_src(config, profile)
         74     elif attribute == IframeAttribute.srcdoc:
    ---> 75         output = get_notebook_iframe_srcdoc(config, profile)
         76     else:
         77         raise ValueError(
    

    D:\Anaonda\lib\site-packages\pandas_profiling\report\presentation\flavours\widget\notebook.py in get_notebook_iframe_srcdoc(config, profile)
         27     width = config.notebook.iframe.width
         28     height = config.notebook.iframe.height
    ---> 29     src = html.escape(profile.to_html())
         30 
         31     iframe = f'<iframe width="{width}" height="{height}" srcdoc="{src}" frameborder="0" allowfullscreen></iframe>'
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in to_html(self)
        386 
        387         """
    --> 388         return self.html
        389 
        390     def to_json(self) -> str:
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in html(self)
        203     def html(self) -> str:
        204         if self._html is None:
    --> 205             self._html = self._render_html()
        206         return self._html
        207 
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in _render_html(self)
        305         from pandas_profiling.report.presentation.flavours import HTMLReport
        306 
    --> 307         report = self.report
        308 
        309         with tqdm(
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in report(self)
        197     def report(self) -> Root:
        198         if self._report is None:
    --> 199             self._report = get_report_structure(self.config, self.description_set)
        200         return self._report
        201 
    

    D:\Anaonda\lib\site-packages\pandas_profiling\profile_report.py in description_set(self)
        179     def description_set(self) -> Dict[str, Any]:
        180         if self._description_set is None:
    --> 181             self._description_set = describe_df(
        182                 self.config,
        183                 self.df,
    

    D:\Anaonda\lib\site-packages\pandas_profiling\model\describe.py in describe(config, df, summarizer, typeset, sample)
         93         pbar.total += len(correlation_names)
         94 
    ---> 95         correlations = {
         96             correlation_name: progress(
         97                 calculate_correlation, pbar, f"Calculate {correlation_name} correlation"
    

    D:\Anaonda\lib\site-packages\pandas_profiling\model\describe.py in <dictcomp>(.0)
         94 
         95         correlations = {
    ---> 96             correlation_name: progress(
         97                 calculate_correlation, pbar, f"Calculate {correlation_name} correlation"
         98             )(config, df, correlation_name, series_description)
    

    D:\Anaonda\lib\site-packages\pandas_profiling\utils\progress_bar.py in inner(*args, **kwargs)
          9     def inner(*args, **kwargs) -> Any:
         10         bar.set_postfix_str(message)
    ---> 11         ret = fn(*args, **kwargs)
         12         bar.update()
         13         return ret
    

    D:\Anaonda\lib\site-packages\pandas_profiling\model\correlations.py in calculate_correlation(config, df, correlation_name, summary)
        105     correlation = None
        106     try:
    --> 107         correlation = correlation_measures[correlation_name].compute(
        108             config, df, summary
        109         )
    

    D:\Anaonda\lib\site-packages\multimethod\__init__.py in __call__(self, *args, **kwargs)
        313         func = self[tuple(func(arg) for func, arg in zip(self.type_checkers, args))]
        314         try:
    --> 315             return func(*args, **kwargs)
        316         except TypeError as ex:
        317             raise DispatchError(f"Function {func.__code__}") from ex
    

    D:\Anaonda\lib\site-packages\pandas_profiling\model\pandas\correlations_pandas.py in pandas_phik_compute(config, df, summary)
        152         from phik import phik_matrix
        153 
    --> 154         correlation = phik_matrix(df[selected_cols], interval_cols=list(intcols))
        155 
        156     return correlation
    

    D:\Anaonda\lib\site-packages\phik\phik.py in phik_matrix(df, interval_cols, bins, quantile, noise_correction, dropna, drop_underflow, drop_overflow, verbose, njobs)
        254         verbose=verbose,
        255     )
    --> 256     return phik_from_rebinned_df(
        257         data_binned,
        258         noise_correction,
    

    D:\Anaonda\lib\site-packages\phik\phik.py in phik_from_rebinned_df(data_binned, noise_correction, dropna, drop_underflow, drop_overflow, njobs)
        164         ]
        165     else:
    --> 166         phik_list = Parallel(n_jobs=njobs)(
        167             delayed(_calc_phik)(co, data_binned[list(co)], noise_correction)
        168             for co in itertools.combinations_with_replacement(
    

    D:\Anaonda\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1054 
       1055             with self._backend.retrieval_context():
    -> 1056                 self.retrieve()
       1057             # Make sure that we get a last message telling us we are done
       1058             elapsed_time = time.time() - self._start_time
    

    D:\Anaonda\lib\site-packages\joblib\parallel.py in retrieve(self)
        933             try:
        934                 if getattr(self._backend, 'supports_timeout', False):
    --> 935                     self._output.extend(job.get(timeout=self.timeout))
        936                 else:
        937                     self._output.extend(job.get())
    

    D:\Anaonda\lib\site-packages\joblib\_parallel_backends.py in wrap_future_result(future, timeout)
        540         AsyncResults.get from multiprocessing."""
        541         try:
    --> 542             return future.result(timeout=timeout)
        543         except CfTimeoutError as e:
        544             raise TimeoutError from e
    

    D:\Anaonda\lib\concurrent\futures\_base.py in result(self, timeout)
        439                     return self.__get_result()
        440 
    --> 441                 self._condition.wait(timeout)
        442 
        443                 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
    

    D:\Anaonda\lib\threading.py in wait(self, timeout)
        310         try:    # restore state no matter what (e.g., KeyboardInterrupt)
        311             if timeout is None:
    --> 312                 waiter.acquire()
        313                 gotit = True
        314             else:
    

    KeyboardInterrupt: 



```python

#df = df[~df.index.duplicated(keep="first")]
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 52391 entries, 2016-01-01 00:00:00+01:00 to 2021-12-31 23:00:00+01:00
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   NO      52391 non-null  int64  
     1   NO2     52391 non-null  int64  
     2   PM10    52391 non-null  int64  
     3   CO2     52391 non-null  int64  
     4   TEMP    52391 non-null  float64
     5   HUMI    52391 non-null  float64
    dtypes: float64(2), int64(4)
    memory usage: 2.8+ MB
    


```python
print(np.sum(df.isna()))
```


```python
df['NON']= df['NON'].astype(float)
df['NO2']=df['NO2'].astype(float)
df['PM10']=df['PM10'].astype(float)
df['CO2']=df['CO2'].astype(float)
df['TEMP']=df['TEMP'].astype(float)
df['HUMI']=df['HUMI'].astype(float)

```


```python

```


```python

```


```python

```


```python

```


```python
df['CO2'].plot(figsize=(12,6))
```




    <AxesSubplot:xlabel='DATE/HEURE'>




    
![png](output_14_1.png)
    



```python
#RNN

results= seasonal_decompose(df['CO2'], extrapolate_trend='freq', period=1)
results.plot();


```


    
![png](output_15_0.png)
    



```python
df2= df[['CO2']]
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 52391 entries, 2016-01-01 00:00:00+01:00 to 2021-12-31 23:00:00+01:00
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   NO      52391 non-null  int64  
     1   NO2     52391 non-null  int64  
     2   PM10    52391 non-null  int64  
     3   CO2     52391 non-null  int64  
     4   TEMP    52391 non-null  float64
     5   HUMI    52391 non-null  float64
    dtypes: float64(2), int64(4)
    memory usage: 2.8+ MB
    


```python


scaler= MinMaxScaler()
```


```python
train_data = df2.iloc[:-8760]
test_data = df2.iloc[-8760:]
train_data.info()

scaler.fit(train_data)
scaled_train= scaler.transform(train_data)
scaled_test= scaler.transform(test_data)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 43631 entries, 2016-01-01 00:00:00+01:00 to 2020-12-24 23:00:00+01:00
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   CO2     43631 non-null  int64
    dtypes: int64(1)
    memory usage: 681.7+ KB
    


```python
n_input= 8760
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
```


```python

```


```python
#definition du model 
model = Sequential()
model.add(LSTM(1000,input_shape=(n_input,n_features)))
model.add(Dense(1))
#model.compile()
model.compile(optimizer='adam', loss='mse')
```


```python
model.summary()
```

    Model: "sequential_10"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_10 (LSTM)              (None, 1000)              4008000   
                                                                     
     dense_3 (Dense)             (None, 1)                 1001      
                                                                     
    =================================================================
    Total params: 4,009,001
    Trainable params: 4,009,001
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(generator, epochs=0)
```




    <keras.callbacks.History at 0x2ac9170d820>




```python
last_train_group = scaled_train[-8760:]
```


```python
model.predict(last_train_group)
```

    WARNING:tensorflow:Model was constructed with shape (None, 8760, 1) for input KerasTensor(type_spec=TensorSpec(shape=(None, 8760, 1), dtype=tf.float32, name='lstm_10_input'), name='lstm_10_input', description="created by layer 'lstm_10_input'"), but it was called on an input with incompatible shape (None, 1, 1).
    274/274 [==============================] - 3s 5ms/step
    




    array([[0.06331902],
           [0.06261747],
           [0.06261747],
           ...,
           [0.06378675],
           [0.0624304 ],
           [0.06186918]], dtype=float32)




```python
scaled_test[0]
```




    array([0.25132743])




```python
predictions_tests=[]

premier_groupe = scaled_train[-n_input:]
groupe_actuel= premier_groupe.reshape((1, n_input, n_features))

for i in range(len(test_data)):
    #prediction du premier groupe
    groupe_actuel= model.predict(groupe_actuel)[0]
    
    #ajouter la prediction dans la liste
    predictions_tests.append(groupe_actuel)
    
    #on utilise la prediction pour re-initialiser le groupe et enlever la premiere valeur
    groupe_actuel= np.append(groupe_actuel[:,1:,:], [[groupe_actuel]], axis=1)
```

    1/1 [==============================] - 12s 12s/step
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_27004\1562670248.py in <module>
         12 
         13     #on utilise la prediction pour re-initialiser le groupe et enlever la premiere valeur
    ---> 14     groupe_actuel= np.append(groupe_actuel[:,1:,:], [[groupe_actuel]], axis=1)
    

    IndexError: too many indices for array: array is 1-dimensional, but 3 were indexed



```python
predictions_tests_retransforme= scaler.inverse_transform(predictions_tests)
```


```python
test_data['predictions']=predictions_tests_retransforme
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_27004\3083792555.py in <module>
    ----> 1 test_data['predictions']=predictions_tests_retransforme
    

    D:\Anaonda\lib\site-packages\pandas\core\frame.py in __setitem__(self, key, value)
       3975         else:
       3976             # set column
    -> 3977             self._set_item(key, value)
       3978 
       3979     def _setitem_slice(self, key: slice, value):
    

    D:\Anaonda\lib\site-packages\pandas\core\frame.py in _set_item(self, key, value)
       4169         ensure homogeneity.
       4170         """
    -> 4171         value = self._sanitize_column(value)
       4172 
       4173         if (
    

    D:\Anaonda\lib\site-packages\pandas\core\frame.py in _sanitize_column(self, value)
       4902 
       4903         if is_list_like(value):
    -> 4904             com.require_length_match(value, self.index)
       4905         return sanitize_array(value, self.index, copy=True, allow_2d=True)
       4906 
    

    D:\Anaonda\lib\site-packages\pandas\core\common.py in require_length_match(data, index)
        559     """
        560     if len(data) != len(index):
    --> 561         raise ValueError(
        562             "Length of values "
        563             f"({len(data)}) "
    

    ValueError: Length of values (1) does not match length of index (8760)



```python
test_data.plot(figsize=(12,6))
```




    <AxesSubplot:xlabel='DATE/HEURE'>




    
![png](output_30_1.png)
    



```python
rmse= sqrt(mean_squared_error(test_data['CO2'], test_data['predictions']))
print(rmse)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3802             try:
    -> 3803                 return self._engine.get_loc(casted_key)
       3804             except KeyError as err:
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'predictions'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_27004\420152892.py in <module>
    ----> 1 rmse= sqrt(mean_squared_error(test_data['CO2'], test_data['predictions']))
          2 print(rmse)
    

    D:\Anaonda\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       3802             if self.columns.nlevels > 1:
       3803                 return self._getitem_multilevel(key)
    -> 3804             indexer = self.columns.get_loc(key)
       3805             if is_integer(indexer):
       3806                 indexer = [indexer]
    

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3803                 return self._engine.get_loc(casted_key)
       3804             except KeyError as err:
    -> 3805                 raise KeyError(key) from err
       3806             except TypeError:
       3807                 # If we have a listlike key, _check_indexing_error will raise
    

    KeyError: 'predictions'



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
#modèle autoregression
```


```python
dftest = adfuller(df['CO2'], autolag = "AIC")
print('ADF:', dftest[0])
print('P-Value:', dftest[1])
print("Num of lags:", dftest[2])
print('Num of observation used for ADF regression and critical values calculation:', dftest[3])
print('Critical Values:')
for key, val in dftest[4].items():
    print("\t", key, ": ", val)
```

    ADF: -7.2598211499700325
    P-Value: 1.6930124032892653e-10
    Num of lags: 58
    Num of observation used for ADF regression and critical values calculation: 52332
    Critical Values:
    	 1% :  -3.43047496409059
    	 5% :  -2.8615952316158593
    	 10% :  -2.5667993979530026
    


```python
pacf=plot_pacf(df['CO2'], lags=25)
acf= plot_acf(df['CO2'], lags=25)
```


    
![png](output_56_0.png)
    



    
![png](output_56_1.png)
    



```python
X=df['CO2'].values
train=X[:len(X)-365]
test=X[len(X)-365:]
```


```python
model= AutoReg(train, lags=10).fit()
print(model.summary())
```

                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                52026
    Model:                    AutoReg(10)   Log Likelihood             -290800.251
    Method:               Conditional MLE   S.D. of innovations             64.823
    Date:                Mon, 07 Nov 2022   AIC                         581624.502
    Time:                        01:05:33   BIC                         581730.814
    Sample:                            10   HQIC                        581657.743
                                    52026                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         10.6949      0.551     19.399      0.000       9.614      11.776
    y.L1           0.9901      0.004    225.957      0.000       0.982       0.999
    y.L2          -0.0185      0.006     -3.000      0.003      -0.031      -0.006
    y.L3          -0.0644      0.006    -10.470      0.000      -0.077      -0.052
    y.L4           0.0401      0.006      6.508      0.000       0.028       0.052
    y.L5           0.0212      0.006      3.443      0.001       0.009       0.033
    y.L6          -0.0313      0.006     -5.078      0.000      -0.043      -0.019
    y.L7          -0.0611      0.006     -9.927      0.000      -0.073      -0.049
    y.L8           0.0483      0.006      7.845      0.000       0.036       0.060
    y.L9           0.0837      0.006     13.596      0.000       0.072       0.096
    y.L10         -0.0346      0.004     -7.888      0.000      -0.043      -0.026
                                        Roots                                     
    ==============================================================================
                       Real          Imaginary           Modulus         Frequency
    ------------------------------------------------------------------------------
    AR.1            -1.3605           -0.4472j            1.4321           -0.4495
    AR.2            -1.3605           +0.4472j            1.4321            0.4495
    AR.3            -0.7218           -1.1660j            1.3714           -0.3382
    AR.4            -0.7218           +1.1660j            1.3714            0.3382
    AR.5             0.3202           -1.2374j            1.2781           -0.2097
    AR.6             0.3202           +1.2374j            1.2781            0.2097
    AR.7             1.0214           -0.0000j            1.0214           -0.0000
    AR.8             1.1266           -0.6429j            1.2972           -0.0825
    AR.9             1.1266           +0.6429j            1.2972            0.0825
    AR.10            2.6713           -0.0000j            2.6713           -0.0000
    ------------------------------------------------------------------------------
    


```python
print(len(train))
```

    52026
    


```python
pred=model.predict(start=len(train), end=len(X)-1, dynamic=False)
```


```python
plt.plot(pred)
plt.plot(test, color='red')
print(pred)
```

    [ 10.69493863  21.28445168  31.57199154  40.87332086  49.6389489
      58.13426228  66.08671573  72.8451259   78.93857696  85.34799566
      91.81593452  98.29142415 104.63650737 110.89381062 117.1385814
     123.31951203 129.30362348 135.08561281 140.72873675 146.25301149
     151.6640297  156.94912962 162.11606446 167.1851662  172.16291752
     177.04036789 181.81307619 186.48387078 191.05719184 195.53550427
     199.91829296 204.20668744 208.40432735 212.51461232 216.53930064
     220.47959609 224.3370789  228.11378062 231.8115937  235.43193562
     238.97615822 242.44585149 245.84271301 249.16833179 252.42416455
     255.61163605 258.73220737 261.78732896 264.7783785  267.70667462
     270.5735185  273.38020471 276.12800396 278.81814986 281.45184484
     284.03027189 286.5545952  289.02595336 291.4454571  293.81419263
     296.13322509 298.40359848 300.6263342  302.80243112 304.93286719
     307.01860056 309.06056959 311.05969268 313.01686869 314.93297769
     316.80888147 318.64542377 320.44343049 322.20371014 323.92705431
     325.61423803 327.26602006 328.88314323 330.46633474 332.01630657
     333.53375578 335.01936478 336.47380167 337.89772052 339.29176167
     340.65655205 341.99270542 343.30082266 344.58149204 345.8352895
     347.06277888 348.26451219 349.44102986 350.592861   351.72052359
     352.82452475 353.90536096 354.96351829 355.9994726  357.01368978
     358.00662592 358.97872756 359.93043186 360.86216682 361.77435145
     362.66739596 363.54170197 364.39766266 365.23566297 366.05607975
     366.85928196 367.6456308  368.41547991 369.16917549 369.90705648
     370.62945473 371.33669509 372.02909563 372.70696772 373.37061622
     374.02033956 374.65642994 375.27917342 375.88885004 376.48573399
     377.07009369 377.64219193 378.202286   378.75062778 379.28746385
     379.81303564 380.3275795  380.83132684 381.32450418 381.80733333
     382.2800314  382.74281099 383.1958802  383.63944279 384.07369822
     384.4988418  384.91506471 385.32255413 385.72149331 386.11206167
     386.49443484 386.86878478 387.23527984 387.59408484 387.94536113
     388.28926669 388.62595618 388.95558101 389.27828941 389.59422651
     389.90353439 390.20635215 390.50281597 390.79305918 391.07721229
     391.3554031  391.62775671 391.89439559 392.15543967 392.41100634
     392.66121052 392.90616474 393.14597915 393.3807616  393.61061768
     393.83565075 394.05596202 394.27165056 394.48281336 394.6895454
     394.89193964 395.09008709 395.28407687 395.47399622 395.65993055
     395.84196346 396.02017683 396.1946508  396.36546383 396.53269274
     396.69641274 396.85669744 397.01361893 397.16724778 397.31765309
     397.46490248 397.60906217 397.75019701 397.88837045 398.02364464
     398.1560804  398.28573731 398.41267366 398.53694654 398.65861184
     398.77772427 398.89433739 399.00850366 399.1202744  399.2296999
     399.33682934 399.44171093 399.54439181 399.64491816 399.7433352
     399.83968719 399.93401744 400.02636839 400.11678157 400.20529763
     400.29195639 400.3767968  400.45985704 400.54117445 400.62078559
     400.69872628 400.77503156 400.84973575 400.92287244 400.99447452
     401.0645742  401.133203   401.20039177 401.26617075 401.3305695
     401.39361699 401.45534157 401.51577101 401.57493247 401.63285256
     401.68955733 401.74507228 401.79942237 401.85263206 401.90472525
     401.9557254  402.00565542 402.05453777 402.10239444 402.14924694
     402.19511635 402.2400233  402.28398798 402.32703016 402.36916919
     402.41042404 402.45081325 402.49035498 402.52906701 402.56696677
     402.60407128 402.64039724 402.67596099 402.7107785  402.74486546
     402.77823717 402.81090866 402.84289461 402.87420941 402.90486714
     402.93488158 402.96426624 402.99303433 403.02119879 403.04877228
     403.07576721 403.1021957  403.12806966 403.15340071 403.17820024
     403.20247942 403.22624915 403.24952013 403.27230281 403.29460746
     403.31644409 403.33782253 403.3587524  403.3792431  403.39930385
     403.41894366 403.43817139 403.45699566 403.47542495 403.49346754
     403.51113155 403.52842492 403.54535542 403.56193068 403.57815815
     403.59404512 403.60959873 403.62482599 403.63973375 403.65432869
     403.6686174  403.68260628 403.69630165 403.70970964 403.72283631
     403.73568753 403.74826911 403.76058669 403.77264581 403.7844519
     403.79601026 403.8073261  403.81840451 403.82925045 403.83986882
     403.85026439 403.86044183 403.87040572 403.88016054 403.88971068
     403.89906043 403.908214   403.9171755  403.92594896 403.93453833
     403.94294747 403.95118017 403.95924012 403.96713096 403.97485622
     403.98241938 403.98982385 403.99707296 404.00416995 404.01111804
     404.01792033 404.02457989 404.03109972 404.03748274 404.04373183
     404.0498498  404.0558394  404.06170331 404.06744419 404.07306461
     404.07856709 404.08395412 404.08922812 404.09439145 404.09944644
     404.10439536 404.10924044 404.11398386 404.11862775 404.1231742
     404.12762525 404.1319829  404.13624912 404.14042583 404.14451489
     404.14851816 404.15243742 404.15627445 404.16003097 404.16370866]
    


    
![png](output_61_1.png)
    



```python
#Calculate error
from math import sqrt
from sklearn.metrics import mean_squared_error


rmse= sqrt(mean_squared_error(test, pred))
```


```python
print(rmse)
```

    184.10671604570706
    


```python
pred_future=model.predict(start=len(X)+1, end=len(X)+365, dynamic=False)
print("the future prediction for the next year")
print(pred_future)
print('Number of predictions made: \t', len(pred_future))
```

    the future prediction for the next year
    [404.17083417 404.17428518 404.17766378 404.1809715  404.1842098
     404.18738016 404.19048399 404.1935227  404.19649765 404.19941017
     404.20226159 404.20505317 404.20778618 404.21046184 404.21308135
     404.21564591 404.21815665 404.22061471 404.22302119 404.22537718
     404.22768374 404.22994189 404.23215267 404.23431705 404.23643602
     404.23851053 404.24054151 404.24252988 404.24447652 404.24638232
     404.24824813 404.25007479 404.25186312 404.25361393 404.255328
     404.2570061  404.25864899 404.26025741 404.26183209 404.26337372
     404.264883   404.26636061 404.26780722 404.26922348 404.27061002
     404.27196747 404.27329643 404.27459751 404.27587128 404.27711833
     404.27833922 404.27953449 404.28070467 404.2818503  404.2829719
     404.28406996 404.28514498 404.28619744 404.28722782 404.28823658
     404.28922417 404.29019104 404.29113763 404.29206435 404.29297162
     404.29385986 404.29472946 404.29558082 404.29641431 404.29723031
     404.29802919 404.29881131 404.29957701 404.30032665 404.30106056
     404.30177907 404.30248251 404.30317118 404.30384541 404.30450548
     404.30515171 404.30578438 404.30640377 404.30701017 404.30760384
     404.30818505 404.30875407 404.30931115 404.30985654 404.31039049
     404.31091323 404.31142501 404.31192605 404.31241657 404.3128968
     404.31336695 404.31382724 404.31427787 404.31471905 404.31515097
     404.31557382 404.31598781 404.3163931  404.3167899  404.31717836
     404.31755868 404.31793101 404.31829554 404.31865241 404.3190018
     404.31934385 404.31967873 404.32000658 404.32032755 404.32064179
     404.32094943 404.32125062 404.32154549 404.32183417 404.32211679
     404.32239349 404.32266438 404.32292958 404.32318922 404.32344341
     404.32369227 404.3239359  404.32417443 404.32440795 404.32463657
     404.32486039 404.32507951 404.32529404 404.32550407 404.32570969
     404.32591099 404.32610807 404.32630102 404.32648992 404.32667485
     404.3268559  404.32703316 404.32720669 404.32737659 404.32754292
     404.32770575 404.32786518 404.32802125 404.32817406 404.32832365
     404.32847011 404.32861349 404.32875387 404.3288913  404.32902584
     404.32915757 404.32928653 404.32941278 404.32953638 404.32965739
     404.32977587 404.32989185 404.3300054  404.33011657 404.33022541
     404.33033196 404.33043628 404.33053841 404.33063839 404.33073628
     404.33083211 404.33092594 404.33101779 404.33110772 404.33119576
     404.33128195 404.33136633 404.33144895 404.33152983 404.33160901
     404.33168653 404.33176243 404.33183673 404.33190947 404.33198069
     404.33205041 404.33211867 404.3321855  404.33225092 404.33231497
     404.33237768 404.33243907 404.33249918 404.33255802 404.33261563
     404.33267203 404.33272725 404.3327813  404.33283423 404.33288604
     404.33293677 404.33298643 404.33303505 404.33308265 404.33312925
     404.33317487 404.33321953 404.33326326 404.33330607 404.33334798
     404.33338902 404.33342919 404.33346852 404.33350702 404.33354472
     404.33358162 404.33361775 404.33365312 404.33368776 404.33372166
     404.33375485 404.33378735 404.33381916 404.33385031 404.3338808
     404.33391065 404.33393988 404.33396849 404.3339965  404.33402393
     404.33405078 404.33407707 404.3341028  404.334128   404.33415266
     404.33417681 404.33420045 404.3342236  404.33424626 404.33426844
     404.33429016 404.33431142 404.33433224 404.33435262 404.33437258
     404.33439211 404.33441123 404.33442996 404.33444829 404.33446623
     404.3344838  404.334501   404.33451784 404.33453433 404.33455047
     404.33456627 404.33458174 404.33459688 404.33461171 404.33462623
     404.33464044 404.33465435 404.33466797 404.33468131 404.33469437
     404.33470715 404.33471966 404.33473191 404.33474391 404.33475565
     404.33476715 404.3347784  404.33478942 404.33480021 404.33481077
     404.33482111 404.33483123 404.33484114 404.33485084 404.33486034
     404.33486964 404.33487875 404.33488766 404.33489639 404.33490493
     404.33491329 404.33492148 404.3349295  404.33493735 404.33494503
     404.33495255 404.33495992 404.33496713 404.33497419 404.3349811
     404.33498786 404.33499449 404.33500097 404.33500732 404.33501354
     404.33501962 404.33502558 404.33503141 404.33503712 404.33504271
     404.33504818 404.33505354 404.33505879 404.33506392 404.33506895
     404.33507387 404.33507869 404.33508341 404.33508803 404.33509255
     404.33509698 404.33510131 404.33510555 404.33510971 404.33511378
     404.33511776 404.33512166 404.33512547 404.33512921 404.33513287
     404.33513645 404.33513995 404.33514339 404.33514675 404.33515004
     404.33515326 404.33515641 404.3351595  404.33516252 404.33516548
     404.33516838 404.33517121 404.33517399 404.33517671 404.33517937
     404.33518197 404.33518452 404.33518702 404.33518947 404.33519186
     404.3351942  404.3351965  404.33519874 404.33520094 404.33520309
     404.3352052  404.33520727 404.33520929 404.33521126 404.3352132
     404.3352151  404.33521695 404.33521877 404.33522055 404.33522229
     404.33522399 404.33522566 404.3352273  404.3352289  404.33523046]
    Number of predictions made: 	 365
    


```python

```


```python
#model ARIMA
```


```python
from pmdarima import auto_arima

stepwise_fit = auto_arima(df['CO2'], trace=True)
stepwise_fit.summary()
```

    Performing stepwise search to minimize aic
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=25632.500, Time=1.74 sec
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=25936.225, Time=0.04 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=25823.768, Time=0.10 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=25760.276, Time=0.17 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=25934.226, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=25635.611, Time=0.91 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=25635.646, Time=0.93 sec
     ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=25634.501, Time=1.54 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=25634.494, Time=2.38 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=25634.321, Time=0.61 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=25637.227, Time=1.46 sec
     ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=25637.432, Time=1.25 sec
     ARIMA(3,1,3)(0,0,0)[0] intercept   : AIC=25622.536, Time=3.61 sec
     ARIMA(4,1,3)(0,0,0)[0] intercept   : AIC=25618.407, Time=2.34 sec
     ARIMA(4,1,2)(0,0,0)[0] intercept   : AIC=25634.348, Time=2.22 sec
     ARIMA(5,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=6.46 sec
     ARIMA(4,1,4)(0,0,0)[0] intercept   : AIC=25608.191, Time=5.44 sec
     ARIMA(3,1,4)(0,0,0)[0] intercept   : AIC=25619.076, Time=4.78 sec
     ARIMA(5,1,4)(0,0,0)[0] intercept   : AIC=25619.835, Time=4.33 sec
     ARIMA(4,1,5)(0,0,0)[0] intercept   : AIC=25617.223, Time=4.73 sec
     ARIMA(3,1,5)(0,0,0)[0] intercept   : AIC=25615.850, Time=4.34 sec
     ARIMA(5,1,5)(0,0,0)[0] intercept   : AIC=25616.373, Time=5.28 sec
     ARIMA(4,1,4)(0,0,0)[0]             : AIC=25606.222, Time=2.60 sec
     ARIMA(3,1,4)(0,0,0)[0]             : AIC=25617.023, Time=2.18 sec
     ARIMA(4,1,3)(0,0,0)[0]             : AIC=25615.973, Time=1.87 sec
     ARIMA(5,1,4)(0,0,0)[0]             : AIC=25617.840, Time=2.36 sec
     ARIMA(4,1,5)(0,0,0)[0]             : AIC=25615.227, Time=2.50 sec
     ARIMA(3,1,3)(0,0,0)[0]             : AIC=25620.552, Time=1.82 sec
     ARIMA(3,1,5)(0,0,0)[0]             : AIC=25613.854, Time=2.31 sec
     ARIMA(5,1,3)(0,0,0)[0]             : AIC=25608.625, Time=2.63 sec
     ARIMA(5,1,5)(0,0,0)[0]             : AIC=25614.379, Time=2.71 sec
    
    Best model:  ARIMA(4,1,4)(0,0,0)[0]          
    Total fit time: 75.720 seconds
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>2187</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(4, 1, 4)</td> <th>  Log Likelihood     </th> <td>-12794.111</td>
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 07 Nov 2022</td> <th>  AIC                </th>  <td>25606.222</td>
</tr>
<tr>
  <th>Time:</th>                <td>00:38:15</td>     <th>  BIC                </th>  <td>25657.431</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th>  <td>25624.940</td>
</tr>
<tr>
  <th></th>                      <td> - 2187</td>     <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>   -0.8555</td> <td>    0.028</td> <td>  -30.421</td> <td> 0.000</td> <td>   -0.911</td> <td>   -0.800</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.6245</td> <td>    0.027</td> <td>  -23.286</td> <td> 0.000</td> <td>   -0.677</td> <td>   -0.572</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.1695</td> <td>    0.030</td> <td>   -5.647</td> <td> 0.000</td> <td>   -0.228</td> <td>   -0.111</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>    0.5438</td> <td>    0.018</td> <td>   30.031</td> <td> 0.000</td> <td>    0.508</td> <td>    0.579</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>    0.5268</td> <td>    0.028</td> <td>   19.010</td> <td> 0.000</td> <td>    0.473</td> <td>    0.581</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    0.1794</td> <td>    0.022</td> <td>    8.101</td> <td> 0.000</td> <td>    0.136</td> <td>    0.223</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>   -0.2683</td> <td>    0.024</td> <td>  -11.220</td> <td> 0.000</td> <td>   -0.315</td> <td>   -0.221</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>   -0.8188</td> <td>    0.022</td> <td>  -37.232</td> <td> 0.000</td> <td>   -0.862</td> <td>   -0.776</td>
</tr>
<tr>
  <th>sigma2</th> <td> 7205.0169</td> <td>   81.372</td> <td>   88.545</td> <td> 0.000</td> <td> 7045.532</td> <td> 7364.502</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.01</td> <th>  Jarque-Bera (JB):  </th> <td>29263.69</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.94</td> <th>  Prob(JB):          </th>   <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.24</td> <th>  Skew:              </th>   <td>-1.23</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>   <td>20.76</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
model=ARIMA(train_data, order=(4, 1, 4))
model=model.fit()
model.summary()
```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>CO2</td>       <th>  No. Observations:  </th>    <td>43631</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(4, 1, 4)</td>  <th>  Log Likelihood     </th> <td>-243797.701</td>
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 07 Nov 2022</td> <th>  AIC                </th> <td>487613.402</td> 
</tr>
<tr>
  <th>Time:</th>                <td>01:22:23</td>     <th>  BIC                </th> <td>487691.553</td> 
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>487638.038</td> 
</tr>
<tr>
  <th></th>                     <td> - 43631</td>     <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.4279</td> <td>    0.012</td> <td>   36.442</td> <td> 0.000</td> <td>    0.405</td> <td>    0.451</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.0788</td> <td>    0.016</td> <td>   -5.024</td> <td> 0.000</td> <td>   -0.110</td> <td>   -0.048</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.5293</td> <td>    0.016</td> <td>  -32.244</td> <td> 0.000</td> <td>   -0.562</td> <td>   -0.497</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>    0.8528</td> <td>    0.010</td> <td>   86.264</td> <td> 0.000</td> <td>    0.833</td> <td>    0.872</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.5100</td> <td>    0.012</td> <td>  -42.284</td> <td> 0.000</td> <td>   -0.534</td> <td>   -0.486</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    0.0026</td> <td>    0.018</td> <td>    0.149</td> <td> 0.882</td> <td>   -0.032</td> <td>    0.037</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>    0.4555</td> <td>    0.018</td> <td>   25.695</td> <td> 0.000</td> <td>    0.421</td> <td>    0.490</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>   -0.9048</td> <td>    0.012</td> <td>  -77.504</td> <td> 0.000</td> <td>   -0.928</td> <td>   -0.882</td>
</tr>
<tr>
  <th>sigma2</th> <td> 4335.5141</td> <td>    8.420</td> <td>  514.925</td> <td> 0.000</td> <td> 4319.012</td> <td> 4352.016</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>184.04</td> <th>  Jarque-Bera (JB):  </th> <td>2854877.18</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>0.00</td>  <th>  Prob(JB):          </th>    <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th>  <td>0.84</td>  <th>  Skew:              </th>    <td>0.37</td>   
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.00</td>  <th>  Kurtosis:          </th>    <td>42.62</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
start=len(train_data)
end=len(train_data)+len(test_data)-1
pred1=model.predict(start=len(train_data), end=end,type='levels')
print(pred)
```

    2022-12-01    526.970807
    2022-12-02    515.088035
    2022-12-03    523.060057
    2022-12-04    513.066241
    2022-12-05    520.624110
    2022-12-06    510.941785
    2022-12-07    517.723829
    2022-12-08    509.442593
    2022-12-09    516.390771
    2022-12-10    508.706318
    2022-12-11    514.557527
    2022-12-12    507.422905
    2022-12-13    513.438352
    2022-12-14    507.365810
    2022-12-15    512.644515
    2022-12-16    506.538374
    2022-12-17    511.468473
    2022-12-18    506.456794
    2022-12-19    511.294290
    2022-12-20    506.298071
    2022-12-21    510.316666
    2022-12-22    505.915086
    2022-12-23    510.171874
    2022-12-24    506.243915
    2022-12-25    509.712988
    2022-12-26    505.778877
    2022-12-27    509.270167
    2022-12-28    506.128578
    2022-12-29    509.312004
    2022-12-30    505.956678
    2022-12-31    508.696969
    Freq: D, Name: Predication par ARIMA, dtype: float64
    

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:834: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


```python
pred1.plot(legend=True)
test_data['CO2'].plot(legend=True)
```




    <AxesSubplot:xlabel='DATE/HEURE'>




    
![png](output_70_1.png)
    



```python
rmse2=sqrt(mean_squared_error(pred1, test_data))
print(rmse2)
```

    90.18592502542586
    


```python
model2=ARIMA(df['CO2'], order=(4, 1, 4))
model2=model2.fit()
```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    


```python
index_future_dates=pd.date_range(start='2022-12-01', end='2023-12-01')
pred=model2.predict(start=len(df['CO2']),end=len(df['CO2'])+365, type='levels').rename('Predication par ARIMA')
pred.index=index_future_dates
```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:834: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


```python
pred.plot(figsize=(12,5), legends=True)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_1516\410695339.py in <module>
    ----> 1 pred.plot(figsize=(12,5), legends=True)
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_core.py in __call__(self, *args, **kwargs)
        998                     data.columns = label_name
        999 
    -> 1000         return plot_backend.plot(data, kind=kind, **kwargs)
       1001 
       1002     __call__.__doc__ = __doc__
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\__init__.py in plot(data, kind, **kwargs)
         69             kwargs["ax"] = getattr(ax, "left_ax", ax)
         70     plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    ---> 71     plot_obj.generate()
         72     plot_obj.draw()
         73     return plot_obj.result
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\core.py in generate(self)
        450         self._compute_plot_data()
        451         self._setup_subplots()
    --> 452         self._make_plot()
        453         self._add_table()
        454         self._make_legend()
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\core.py in _make_plot(self)
       1380             kwds["label"] = label
       1381 
    -> 1382             newlines = plotf(
       1383                 ax,
       1384                 x,
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\core.py in _ts_plot(self, ax, x, data, style, **kwds)
       1429         ax._plot_data.append((data, self._kind, kwds))
       1430 
    -> 1431         lines = self._plot(ax, data.index, data.values, style=style, **kwds)
       1432         # set date formatter, locators and rescale limits
       1433         format_dateaxis(ax, ax.freq, data.index)
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\core.py in _plot(cls, ax, x, y, style, column_num, stacking_id, **kwds)
       1410             cls._initialize_stacker(ax, stacking_id, len(y))
       1411         y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
    -> 1412         lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
       1413         cls._update_stacker(ax, stacking_id, y)
       1414         return lines
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\converter.py in wrapper(*args, **kwargs)
         94     def wrapper(*args, **kwargs):
         95         with pandas_converters():
    ---> 96             return func(*args, **kwargs)
         97 
         98     return cast(F, wrapper)
    

    D:\Anaonda\lib\site-packages\pandas\plotting\_matplotlib\core.py in _plot(cls, ax, x, y, style, is_errorbar, **kwds)
        885             # prevent style kwarg from going to errorbar, where it is unsupported
        886             args = (x, y, style) if style is not None else (x, y)
    --> 887             return ax.plot(*args, **kwds)
        888 
        889     def _get_custom_index_name(self):
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_axes.py in plot(self, scalex, scaley, data, *args, **kwargs)
       1630         """
       1631         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    -> 1632         lines = [*self._get_lines(*args, data=data, **kwargs)]
       1633         for line in lines:
       1634             self.add_line(line)
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_base.py in __call__(self, data, *args, **kwargs)
        310                 this += args[0],
        311                 args = args[1:]
    --> 312             yield from self._plot_args(this, kwargs)
        313 
        314     def get_next_color(self):
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_base.py in _plot_args(self, tup, kwargs, return_kwargs)
        536             return list(result)
        537         else:
    --> 538             return [l[0] for l in result]
        539 
        540 
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_base.py in <listcomp>(.0)
        536             return list(result)
        537         else:
    --> 538             return [l[0] for l in result]
        539 
        540 
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_base.py in <genexpr>(.0)
        529             labels = [label] * n_datasets
        530 
    --> 531         result = (make_artist(x[:, j % ncx], y[:, j % ncy], kw,
        532                               {**kwargs, 'label': label})
        533                   for j, label in enumerate(labels))
    

    D:\Anaonda\lib\site-packages\matplotlib\axes\_base.py in _makeline(self, x, y, kw, kwargs)
        349         default_dict = self._getdefaults(set(), kw)
        350         self._setdefaults(default_dict, kw)
    --> 351         seg = mlines.Line2D(x, y, **kw)
        352         return seg, kw
        353 
    

    D:\Anaonda\lib\site-packages\matplotlib\lines.py in __init__(self, xdata, ydata, linewidth, linestyle, color, marker, markersize, markeredgewidth, markeredgecolor, markerfacecolor, markerfacecoloralt, fillstyle, antialiased, dash_capstyle, solid_capstyle, dash_joinstyle, solid_joinstyle, pickradius, drawstyle, markevery, **kwargs)
        391         # update kwargs before updating data to give the caller a
        392         # chance to init axes (and hence unit support)
    --> 393         self.update(kwargs)
        394         self.pickradius = pickradius
        395         self.ind_offset = 0
    

    D:\Anaonda\lib\site-packages\matplotlib\artist.py in update(self, props)
       1062                     func = getattr(self, f"set_{k}", None)
       1063                     if not callable(func):
    -> 1064                         raise AttributeError(f"{type(self).__name__!r} object "
       1065                                              f"has no property {k!r}")
       1066                     ret.append(func(v))
    

    AttributeError: 'Line2D' object has no property 'legends'



    
![png](output_74_1.png)
    


plt.figure(figsize=(20,6))
plt.plot(df.groupby('DATE/HEURE').count(), 'o', color='skyblue')
plt.title('Nb of measurements per DATE/HEURE')
plt.ylabel('number of measurements')
plt.xlabel('DATE/HEURE')
plt.show()


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
print(np.sum(df.isna()))
```

    NO      0
    NO2     0
    PM10    0
    CO2     0
    TEMP    0
    HUMI    0
    dtype: int64
    


```python

```

    NON                 0
    NO2                 0
    PM10                0
    CO2                 0
    TEMP                0
    HUMI                0
    First Difference    1
    Second order        2
    dtype: int64
    


```python
df.fillna(0.0)


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NON</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
      <th>First Difference</th>
      <th>Second order</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-31</th>
      <td>32.0</td>
      <td>37.0</td>
      <td>139.0</td>
      <td>649.0</td>
      <td>23.1</td>
      <td>36.1</td>
      <td>120.0</td>
      <td>171.0</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>16.0</td>
      <td>26.0</td>
      <td>108.0</td>
      <td>529.0</td>
      <td>22.0</td>
      <td>37.1</td>
      <td>-51.0</td>
      <td>-109.0</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>10.0</td>
      <td>26.0</td>
      <td>128.0</td>
      <td>580.0</td>
      <td>22.2</td>
      <td>35.6</td>
      <td>58.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>8.0</td>
      <td>22.0</td>
      <td>133.0</td>
      <td>522.0</td>
      <td>21.8</td>
      <td>39.5</td>
      <td>-7.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>15.0</td>
      <td>35.0</td>
      <td>149.0</td>
      <td>529.0</td>
      <td>22.1</td>
      <td>34.6</td>
      <td>-25.0</td>
      <td>-11.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>5.0</td>
      <td>21.0</td>
      <td>33.0</td>
      <td>477.0</td>
      <td>18.3</td>
      <td>59.9</td>
      <td>-1.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>6.0</td>
      <td>19.0</td>
      <td>44.0</td>
      <td>478.0</td>
      <td>17.9</td>
      <td>47.6</td>
      <td>-11.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>6.0</td>
      <td>17.0</td>
      <td>28.0</td>
      <td>489.0</td>
      <td>19.2</td>
      <td>71.3</td>
      <td>-12.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>20.0</td>
      <td>29.0</td>
      <td>26.0</td>
      <td>501.0</td>
      <td>19.4</td>
      <td>63.7</td>
      <td>-18.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>9.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>519.0</td>
      <td>19.1</td>
      <td>62.1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2187 rows × 8 columns</p>
</div>




```python
df['NON']= df['NON'].astype(float)
df['NO2']=df['NO2'].astype(float)
df['PM10']=df['PM10'].astype(float)
df['CO2']=df['CO2'].astype(float)
df['TEMP']=df['TEMP'].astype(float)
df['HUMI']=df['HUMI'].astype(float)

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3802             try:
    -> 3803                 return self._engine.get_loc(casted_key)
       3804             except KeyError as err:
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'NON'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\1226107739.py in <module>
    ----> 1 df['NON']= df['NON'].astype(float)
          2 df['NO2']=df['NO2'].astype(float)
          3 df['PM10']=df['PM10'].astype(float)
          4 df['CO2']=df['CO2'].astype(float)
          5 df['TEMP']=df['TEMP'].astype(float)
    

    D:\Anaonda\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       3802             if self.columns.nlevels > 1:
       3803                 return self._getitem_multilevel(key)
    -> 3804             indexer = self.columns.get_loc(key)
       3805             if is_integer(indexer):
       3806                 indexer = [indexer]
    

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3803                 return self._engine.get_loc(casted_key)
       3804             except KeyError as err:
    -> 3805                 raise KeyError(key) from err
       3806             except TypeError:
       3807                 # If we have a listlike key, _check_indexing_error will raise
    

    KeyError: 'NON'



```python
#df['NON']=df['NON'].replace(0.0,df['CO2'].mean())
#df['NO2']=df['NO2'].replace(0.0,df['CO2'].mean())
#df['PM10']=df['PM10'].replace(0.0,df['CO2'].mean())
#df['TEMP']=df['TEMP'].replace(0.0,df['CO2'].mean())
#df['HUMI']=df['HUMI'].replace(0.0,df['CO2'].mean())

df['CO2']=df['CO2'].replace('<',df['CO2'].mean())
```


```python
# getting the index data
#index = df.index

# removing duplicate indices separately
df = df[~index.duplicated(keep="first")]
df.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NON</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
      <th>First Difference</th>
      <th>Second order</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-31</th>
      <td>32.0</td>
      <td>37.0</td>
      <td>139.0</td>
      <td>649.0</td>
      <td>23.1</td>
      <td>36.1</td>
      <td>120.0</td>
      <td>171.0</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>16.0</td>
      <td>26.0</td>
      <td>108.0</td>
      <td>529.0</td>
      <td>22.0</td>
      <td>37.1</td>
      <td>-51.0</td>
      <td>-109.0</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>10.0</td>
      <td>26.0</td>
      <td>128.0</td>
      <td>580.0</td>
      <td>22.2</td>
      <td>35.6</td>
      <td>58.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>8.0</td>
      <td>22.0</td>
      <td>133.0</td>
      <td>522.0</td>
      <td>21.8</td>
      <td>39.5</td>
      <td>-7.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>15.0</td>
      <td>35.0</td>
      <td>149.0</td>
      <td>529.0</td>
      <td>22.1</td>
      <td>34.6</td>
      <td>-25.0</td>
      <td>-11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

df.duplicated().sum()
```




    409




```python

```


```python

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NON</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
      <th>First Difference</th>
      <th>Second order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2187.000000</td>
      <td>2187.000000</td>
      <td>2187.000000</td>
      <td>2187.000000</td>
      <td>2187.000000</td>
      <td>2187.000000</td>
      <td>2186.000000</td>
      <td>2185.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>169.708647</td>
      <td>157.808776</td>
      <td>180.690735</td>
      <td>466.430054</td>
      <td>107.952809</td>
      <td>128.690500</td>
      <td>0.059469</td>
      <td>0.063158</td>
    </tr>
    <tr>
      <th>std</th>
      <td>213.460153</td>
      <td>197.188879</td>
      <td>168.719592</td>
      <td>69.656488</td>
      <td>177.153138</td>
      <td>167.378756</td>
      <td>39.568673</td>
      <td>60.156227</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>5.000000</td>
      <td>372.700960</td>
      <td>11.400000</td>
      <td>18.900000</td>
      <td>-260.299040</td>
      <td>-447.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000</td>
      <td>25.000000</td>
      <td>58.000000</td>
      <td>372.700960</td>
      <td>18.600000</td>
      <td>40.700000</td>
      <td>-13.000000</td>
      <td>-17.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19.000000</td>
      <td>37.000000</td>
      <td>103.000000</td>
      <td>469.000000</td>
      <td>21.600000</td>
      <td>48.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>466.430054</td>
      <td>466.430054</td>
      <td>229.000000</td>
      <td>512.000000</td>
      <td>26.100000</td>
      <td>62.300000</td>
      <td>8.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>466.430054</td>
      <td>466.430054</td>
      <td>885.000000</td>
      <td>777.000000</td>
      <td>466.430054</td>
      <td>466.430054</td>
      <td>254.000000</td>
      <td>427.598080</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_126_1.png)
    



```python

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NON</th>
      <th>NO2</th>
      <th>PM10</th>
      <th>CO2</th>
      <th>TEMP</th>
      <th>HUMI</th>
      <th>First Difference</th>
      <th>Second order</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-31</th>
      <td>32.0</td>
      <td>37.0</td>
      <td>139.0</td>
      <td>649.0</td>
      <td>23.1</td>
      <td>36.1</td>
      <td>120.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>16.0</td>
      <td>26.0</td>
      <td>108.0</td>
      <td>529.0</td>
      <td>22.0</td>
      <td>37.1</td>
      <td>-51.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>10.0</td>
      <td>26.0</td>
      <td>128.0</td>
      <td>580.0</td>
      <td>22.2</td>
      <td>35.6</td>
      <td>58.0</td>
      <td>171.0</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>8.0</td>
      <td>22.0</td>
      <td>133.0</td>
      <td>522.0</td>
      <td>21.8</td>
      <td>39.5</td>
      <td>-7.0</td>
      <td>-109.0</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>15.0</td>
      <td>35.0</td>
      <td>149.0</td>
      <td>529.0</td>
      <td>22.1</td>
      <td>34.6</td>
      <td>-25.0</td>
      <td>65.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2= df[['CO2']]

df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO2</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-31</th>
      <td>649.0</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>529.0</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>580.0</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>522.0</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>529.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python
df2[df['CO2'] == df2['CO2'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO2</th>
    </tr>
    <tr>
      <th>DATE/HEURE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-10</th>
      <td>777.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\sarimax.py:866: UserWarning: Too few observations to estimate starting parameters for ARMA and trend. All parameters except for variances will be set to zeros.
      warn('Too few observations to estimate starting parameters%s.'
    D:\Anaonda\lib\site-packages\numpy\core\fromnumeric.py:3723: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    D:\Anaonda\lib\site-packages\numpy\core\_methods.py:222: RuntimeWarning: invalid value encountered in true_divide
      arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
    D:\Anaonda\lib\site-packages\numpy\core\_methods.py:254: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\model.py:531: RuntimeWarning: invalid value encountered in double_scalars
      return -self.loglike(params, *args) / nobs
    


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\1981344022.py in <module>
          1 arima = ARIMA(df2.loc[:'2000-01-01','CO2'], order=(28, 1 ,0))
    ----> 2 results = arima.fit()
          3 predictions = results.predict('2000-01-01', '2014-01-01')
          4 
          5 plt.plot(df.loc['2000-01-01':,'CO2'],label='Actual')
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\arima\model.py in fit(self, start_params, transformed, includes_fixed, method, method_kwargs, gls, gls_kwargs, cov_type, cov_kwds, return_params, low_memory)
        388                 method_kwargs.setdefault('disp', 0)
        389 
    --> 390                 res = super().fit(
        391                     return_params=return_params, low_memory=low_memory,
        392                     cov_type=cov_type, cov_kwds=cov_kwds, **method_kwargs)
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\mlemodel.py in fit(self, start_params, transformed, includes_fixed, cov_type, cov_kwds, method, maxiter, full_output, disp, callback, return_params, optim_score, optim_complex_step, optim_hessian, flags, low_memory, **kwargs)
        702                 flags['hessian_method'] = optim_hessian
        703             fargs = (flags,)
    --> 704             mlefit = super(MLEModel, self).fit(start_params, method=method,
        705                                                fargs=fargs,
        706                                                maxiter=maxiter,
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\model.py in fit(self, start_params, method, maxiter, full_output, disp, fargs, callback, retall, skip_hessian, **kwargs)
        561 
        562         optimizer = Optimizer()
    --> 563         xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
        564                                                        fargs, kwargs,
        565                                                        hessian=hess,
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\optimizer.py in _fit(self, objective, gradient, start_params, fargs, kwargs, hessian, method, maxiter, full_output, disp, callback, retall)
        239 
        240         func = fit_funcs[method]
    --> 241         xopt, retvals = func(objective, gradient, start_params, fargs, kwargs,
        242                              disp=disp, maxiter=maxiter, callback=callback,
        243                              retall=retall, full_output=full_output,
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\optimizer.py in _fit_lbfgs(f, score, start_params, fargs, kwargs, disp, maxiter, callback, retall, full_output, hess)
        649         func = f
        650 
    --> 651     retvals = optimize.fmin_l_bfgs_b(func, start_params, maxiter=maxiter,
        652                                      callback=callback, args=fargs,
        653                                      bounds=bounds, disp=disp,
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_lbfgsb_py.py in fmin_l_bfgs_b(func, x0, fprime, args, approx_grad, bounds, m, factr, pgtol, epsilon, iprint, maxfun, maxiter, disp, callback, maxls)
        197             'maxls': maxls}
        198 
    --> 199     res = _minimize_lbfgsb(fun, x0, args=args, jac=jac, bounds=bounds,
        200                            **opts)
        201     d = {'grad': res['jac'],
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_lbfgsb_py.py in _minimize_lbfgsb(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, finite_diff_rel_step, **unknown_options)
        360             # until the completion of the current minimization iteration.
        361             # Overwrite f and g:
    --> 362             f, g = func_and_grad(x)
        363         elif task_str.startswith(b'NEW_X'):
        364             # new iteration
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_differentiable_functions.py in fun_and_grad(self, x)
        283         if not np.array_equal(x, self.x):
        284             self._update_x_impl(x)
    --> 285         self._update_fun()
        286         self._update_grad()
        287         return self.f, self.g
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_differentiable_functions.py in _update_fun(self)
        249     def _update_fun(self):
        250         if not self.f_updated:
    --> 251             self._update_fun_impl()
        252             self.f_updated = True
        253 
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_differentiable_functions.py in update_fun()
        153 
        154         def update_fun():
    --> 155             self.f = fun_wrapped(self.x)
        156 
        157         self._update_fun_impl = update_fun
    

    D:\Anaonda\lib\site-packages\scipy\optimize\_differentiable_functions.py in fun_wrapped(x)
        135             # Overwriting results in undefined behaviour because
        136             # fun(self.x) will change self.x, with the two no longer linked.
    --> 137             fx = fun(np.copy(x), *args)
        138             # Make sure the function returns a true scalar
        139             if not np.isscalar(fx):
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\model.py in f(params, *args)
        529 
        530         def f(params, *args):
    --> 531             return -self.loglike(params, *args) / nobs
        532 
        533         if method == 'newton':
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\mlemodel.py in loglike(self, params, *args, **kwargs)
        937             kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        938 
    --> 939         loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
        940 
        941         # Koopman, Shephard, and Doornik recommend maximizing the average
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\kalman_filter.py in loglike(self, **kwargs)
        981         kwargs.setdefault('conserve_memory',
        982                           MEMORY_CONSERVE ^ MEMORY_NO_LIKELIHOOD)
    --> 983         kfilter = self._filter(**kwargs)
        984         loglikelihood_burn = kwargs.get('loglikelihood_burn',
        985                                         self.loglikelihood_burn)
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\kalman_filter.py in _filter(self, filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn, complex_step)
        901 
        902         # Initialize the state
    --> 903         self._initialize_state(prefix=prefix, complex_step=complex_step)
        904 
        905         # Run the filter
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\representation.py in _initialize_state(self, prefix, complex_step)
        981             if not self.initialization.initialized:
        982                 raise RuntimeError('Initialization is incomplete.')
    --> 983             self._statespaces[prefix].initialize(self.initialization,
        984                                                  complex_step=complex_step)
        985         else:
    

    statsmodels\tsa\statespace\_representation.pyx in statsmodels.tsa.statespace._representation.dStatespace.initialize()
    

    statsmodels\tsa\statespace\_representation.pyx in statsmodels.tsa.statespace._representation.dStatespace.initialize()
    

    statsmodels\tsa\statespace\_initialization.pyx in statsmodels.tsa.statespace._initialization.dInitialization.initialize()
    

    statsmodels\tsa\statespace\_initialization.pyx in statsmodels.tsa.statespace._initialization.dInitialization.initialize_stationary_stationary_cov()
    

    statsmodels\tsa\statespace\_tools.pyx in statsmodels.tsa.statespace._tools._dsolve_discrete_lyapunov()
    

    LinAlgError: Schur decomposition solver error.



```python

```


```python


plt.figure(figsize=(14,6))
sns.lineplot(data=df2["CO2"], color='red')
plt.title('Quantite de CO2', fontsize=15)
```




    Text(0.5, 1.0, 'Quantite de CO2')




    
![png](output_137_1.png)
    



```python

```


```python
 plt.figure(figsize=(14,2))
sns.boxplot(x = df2['CO2'], palette='Pastel1')
```




    <AxesSubplot:xlabel='CO2'>




    
![png](output_139_1.png)
    



```python
from statsmodels.tsa.seasonal import STL
plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')

stl = STL(df2['CO2'], period=10, robust=True)
res = stl.fit()
res.plot()
```




    
![png](output_140_0.png)
    




    <Figure size 4000x1600 with 0 Axes>



    
![png](output_140_2.png)
    



```python
fig = plt.figure(figsize=(15, 7))
layout = (3, 2)
pm_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
mv_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
fit_ax = plt.subplot2grid(layout, (2, 0), colspan=2)

pm_ax.plot(res.trend)
pm_ax.set_title("Automatic decomposed trend")

mm = df2['CO2'].rolling(10).mean()
mv_ax.plot(mm)
mv_ax.set_title("Moving average 10 steps")


X = [i for i in range(0, len(df['CO2']))]
X = np.reshape(X, (len(X), 1))
y = df2['CO2'].values
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
fit_ax.plot(trend)
fit_ax.set_title("Trend fitted by linear regression")

plt.tight_layout()
```


    
![png](output_141_0.png)
    



```python
adfuller(df2['CO2']) 
```




    (-3.6202487928547225,
     0.005385507282920361,
     26,
     2160,
     {'1%': -3.4333810594081227,
      '5%': -2.862879013318124,
      '10%': -2.5674828242884087},
     21348.124851270397)




```python
kpss(df2['CO2'])
```




    (0.5575367361106237,
     0.02870794231742709,
     27,
     {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})




```python
#transfirmer la time series

df2['First Difference'] = df2['CO2'] - df2['CO2'].shift(1)
plt.plot(df2['First Difference'])
```

    C:\Users\joshn\AppData\Local\Temp\ipykernel_7840\2277442889.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df2['First Difference'] = df2['CO2'] - df2['CO2'].shift(1)
    




    [<matplotlib.lines.Line2D at 0x206c4b01310>]




    
![png](output_144_2.png)
    



```python
plot_acf(df2['First Difference'].dropna(),lags=25)
plot_pacf(df2['First Difference'].dropna(),lags=25)
```




    
![png](output_145_0.png)
    




    
![png](output_145_1.png)
    



    
![png](output_145_2.png)
    



```python
df['Second order'] = df2['CO2'] - 2*df2['CO2'].shift(1) + df2['CO2'].shift(2)
plt.plot(df['Second order'])
```




    [<matplotlib.lines.Line2D at 0x206c4b01ee0>]




    
![png](output_146_1.png)
    



```python
adfuller(df2['First Difference'].dropna())
```




    (-12.811225484915196,
     6.44018984888339e-24,
     26,
     2159,
     {'1%': -3.4333824650008378,
      '5%': -2.862879633943035,
      '10%': -2.5674831547315},
     21335.49732509593)




```python
plt.figure(figsize=(10,4))
plt.plot(np.sqrt(df2['CO2']).diff())
```




    [<matplotlib.lines.Line2D at 0x206c4b6ea00>]




    
![png](output_148_1.png)
    



```python
a,b,c,d,f = np.array_split(df2, 5) # divide the dataset into 5 equal parts 
plt.figure(figsize=(12,12))
plt.subplot(511)
sns.lineplot(data=a['CO2'])

plt.subplot(512)
sns.lineplot(data=b['CO2'])

plt.subplot(513)
sns.lineplot(data=c['CO2'])

plt.subplot(514)
sns.lineplot(data=d['CO2'])

plt.subplot(515)
sns.lineplot(data=f['CO2'])
#we can see in the very begining the values are decrising and then continuesly incresing over the time. in last few years we can see sligh drop in co2 levels.
```




    <AxesSubplot:xlabel='DATE/HEURE', ylabel='CO2'>




    
![png](output_149_1.png)
    



```python
#SQRT transformation

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.hist(np.sqrt(df2['CO2']))
plt.subplot(122)
plt.plot(np.sqrt(df2['CO2']))
```




    [<matplotlib.lines.Line2D at 0x206d16f4580>]




    
![png](output_150_1.png)
    



```python
new_df = df2[df2['CO2']>0.5]
```


```python
new_df.plot()
```




    <AxesSubplot:xlabel='DATE/HEURE'>




    
![png](output_152_1.png)
    



```python
#train et test split

train_data = df.loc[:'2020-12-30','CO2']
train_data.info()
test_data = df2.loc['2021-01-01':]
test_data.info()
```

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 1827 entries, 2015-12-31 to 2020-12-30
    Series name: CO2
    Non-Null Count  Dtype  
    --------------  -----  
    1827 non-null   float64
    dtypes: float64(1)
    memory usage: 28.5 KB
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 359 entries, 2021-01-01 to 2021-12-31
    Data columns (total 2 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   CO2               359 non-null    float64
     1   First Difference  359 non-null    float64
    dtypes: float64(2)
    memory usage: 8.4 KB
    


```python
plt.figure(figsize=(10,5))
plt.plot(train_data, color='red')
```




    [<matplotlib.lines.Line2D at 0x206e6994820>]




    
![png](output_154_1.png)
    



```python
 plt.figure(figsize=(7,5))
plt.plot(test_data['CO2'], color = 'green')
```




    [<matplotlib.lines.Line2D at 0x206e6898ee0>]




    
![png](output_155_1.png)
    



```python

# Instantiate and fit the AR model with training data
ar_model = AutoReg(train_data, lags=15).fit()

# Print Summary
print(ar_model.summary())
```

                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                    CO2   No. Observations:                  361
    Model:                    AutoReg(15)   Log Likelihood               -1548.856
    Method:               Conditional MLE   S.D. of innovations             21.275
    Date:                Sun, 06 Nov 2022   AIC                           3131.712
    Time:                        16:01:21   BIC                           3197.102
    Sample:                            15   HQIC                          3157.751
                                      361                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         44.0076     24.902      1.767      0.077      -4.800      92.816
    CO2.L1         0.4748      0.053      8.888      0.000       0.370       0.579
    CO2.L2        -0.1415      0.057     -2.468      0.014      -0.254      -0.029
    CO2.L3         0.0755      0.058      1.307      0.191      -0.038       0.189
    CO2.L4         0.0269      0.058      0.466      0.641      -0.086       0.140
    CO2.L5         0.0384      0.057      0.671      0.502      -0.074       0.150
    CO2.L6         0.0170      0.057      0.299      0.765      -0.095       0.128
    CO2.L7         0.2705      0.057      4.768      0.000       0.159       0.382
    CO2.L8        -0.0468      0.059     -0.800      0.424      -0.162       0.068
    CO2.L9         0.0984      0.057      1.723      0.085      -0.014       0.210
    CO2.L10        0.0053      0.057      0.093      0.926      -0.107       0.118
    CO2.L11       -0.0580      0.057     -1.015      0.310      -0.170       0.054
    CO2.L12       -0.0084      0.056     -0.149      0.881      -0.119       0.102
    CO2.L13        0.0043      0.056      0.076      0.939      -0.106       0.114
    CO2.L14        0.2689      0.056      4.804      0.000       0.159       0.379
    CO2.L15       -0.1172      0.053     -2.221      0.026      -0.221      -0.014
                                        Roots                                     
    ==============================================================================
                       Real          Imaginary           Modulus         Frequency
    ------------------------------------------------------------------------------
    AR.1            -1.1490           -0.0000j            1.1490           -0.5000
    AR.2            -0.9586           -0.4767j            1.0706           -0.4266
    AR.3            -0.9586           +0.4767j            1.0706            0.4266
    AR.4            -0.6974           -0.9062j            1.1435           -0.3544
    AR.5            -0.6974           +0.9062j            1.1435            0.3544
    AR.6            -0.2395           -1.0357j            1.0631           -0.2862
    AR.7            -0.2395           +1.0357j            1.0631            0.2862
    AR.8             0.2295           -1.0656j            1.0900           -0.2162
    AR.9             0.2295           +1.0656j            1.0900            0.2162
    AR.10            0.6719           -0.8119j            1.0539           -0.1400
    AR.11            0.6719           +0.8119j            1.0539            0.1400
    AR.12            1.0186           -0.0000j            1.0186           -0.0000
    AR.13            1.0690           -0.5399j            1.1976           -0.0744
    AR.14            1.0690           +0.5399j            1.1976            0.0744
    AR.15            2.2747           -0.0000j            2.2747           -0.0000
    ------------------------------------------------------------------------------
    

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    


```python
df2= df2.drop.na
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\109058965.py in <module>
    ----> 1 df2= df2.drop.na
    

    AttributeError: 'function' object has no attribute 'na'



```python
ar_forecast = ar_model.predict('2021-01-01', '2021-12-31')

plt.title('AR Model Results', size=20)
plt.plot(df2.loc['2021-01-01':,'CO2'],label='Actual')
plt.plot(ar_forecast,label='Predicted')
plt.legend();

print('RMSE:', np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'CO2'],ar_forecast)))
print('MAE:', mean_absolute_error(df2.loc['2021-01-01':,'CO2'],ar_forecast))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\3512352651.py in <module>
          6 plt.legend();
          7 
    ----> 8 print('RMSE:', np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'CO2'],ar_forecast)))
          9 print('MAE:', mean_absolute_error(df2.loc['2021-01-01':,'CO2'],ar_forecast))
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
        436     0.825...
        437     """
    --> 438     y_type, y_true, y_pred, multioutput = _check_reg_targets(
        439         y_true, y_pred, multioutput
        440     )
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
         94     check_consistent_length(y_true, y_pred)
         95     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    ---> 96     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
         97 
         98     if y_true.ndim == 1:
    

    D:\Anaonda\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        798 
        799         if force_all_finite:
    --> 800             _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
        801 
        802     if ensure_min_samples > 0:
    

    D:\Anaonda\lib\site-packages\sklearn\utils\validation.py in _assert_all_finite(X, allow_nan, msg_dtype)
        112         ):
        113             type_err = "infinity" if allow_nan else "NaN, infinity"
    --> 114             raise ValueError(
        115                 msg_err.format(
        116                     type_err, msg_dtype if msg_dtype is not None else X.dtype
    

    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').



    
![png](output_158_1.png)
    



```python
df2_2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2187 entries, 2015-12-31 to 2021-12-31
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   CO2     2187 non-null   float64
    dtypes: float64(1)
    memory usage: 98.7 KB
    


```python
train_datafd = df2.loc['2016-01-01':'2020-12-30','First Difference']

# Instantiate and fit the AR model with training data
ar_modelfd = AutoReg(train_datafd, lags=15).fit()

# Print Summary
#print(ar_modelfd.summary())

ar_forecastfd = ar_modelfd.predict('2021-01-01', '2021-12-30')
plt.title('AR Model Results', size=20)
plt.plot(df2.loc['2021-01-01':,'First Difference'],label='Actual')
plt.plot(ar_forecastfd,label='Predicted')
plt.legend();

print('RMSE:',np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'First Difference'],ar_forecastfd)))
print('MAE:',mean_absolute_error(df2.loc['2021-01-01':,'First Difference'],ar_forecastfd))
```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\deterministic.py:302: UserWarning: Only PeriodIndexes, DatetimeIndexes with a frequency set, RangesIndexes, and Index with a unit increment support extending. The index is set will contain the position relative to the data length.
      fcast_index = self._extend_index(index, steps, forecast_index)
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\2327917684.py in <module>
         13 plt.legend();
         14 
    ---> 15 print('RMSE:',np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'First Difference'],ar_forecastfd)))
         16 print('MAE:',mean_absolute_error(df2.loc['2021-01-01':,'First Difference'],ar_forecastfd))
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
        436     0.825...
        437     """
    --> 438     y_type, y_true, y_pred, multioutput = _check_reg_targets(
        439         y_true, y_pred, multioutput
        440     )
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
         92         the dtype argument passed to check_array.
         93     """
    ---> 94     check_consistent_length(y_true, y_pred)
         95     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
         96     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    

    D:\Anaonda\lib\site-packages\sklearn\utils\validation.py in check_consistent_length(*arrays)
        330     uniques = np.unique(lengths)
        331     if len(uniques) > 1:
    --> 332         raise ValueError(
        333             "Found input variables with inconsistent numbers of samples: %r"
        334             % [int(l) for l in lengths]
    

    ValueError: Found input variables with inconsistent numbers of samples: [359, 364]



    
![png](output_160_2.png)
    



```python
arima = ARIMA(train_data, order=(0, 2 ,0))
results = arima.fit()
predictions = results.predict('20-01-01', '2014-01-01')
```

    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(dates, freq)
    C:\Users\joshn\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(dates, freq)
    


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.DatetimeEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()
    

    KeyError: 979948800000000000

    
    During handling of the above exception, another exception occurred:
    

    KeyError                                  Traceback (most recent call last)

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3802             try:
    -> 3803                 return self._engine.get_loc(casted_key)
       3804             except KeyError as err:
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.DatetimeEngine.get_loc()
    

    D:\Anaonda\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.DatetimeEngine.get_loc()
    

    KeyError: Timestamp('2001-01-20 00:00:00')

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    D:\Anaonda\lib\site-packages\pandas\core\indexes\datetimes.py in get_loc(self, key, method, tolerance)
        735         try:
    --> 736             return Index.get_loc(self, key, method, tolerance)
        737         except KeyError as err:
    

    D:\Anaonda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3804             except KeyError as err:
    -> 3805                 raise KeyError(key) from err
       3806             except TypeError:
    

    KeyError: Timestamp('2001-01-20 00:00:00')

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in get_prediction_index(start, end, nobs, base_index, index, silent, index_none, index_generated, data)
        355     try:
    --> 356         start, _, start_oos = get_index_label_loc(
        357             start, base_index, data.row_labels
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in get_index_label_loc(key, index, row_labels)
        278         except:
    --> 279             raise e
        280     return loc, index, index_was_expanded
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in get_index_label_loc(key, index, row_labels)
        242     try:
    --> 243         loc, index, index_was_expanded = get_index_loc(key, index)
        244     except KeyError as e:
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in get_index_loc(key, index)
        175         # (note that get_loc will throw a KeyError if key is invalid)
    --> 176         loc = index.get_loc(key)
        177     elif int_index or range_index:
    

    D:\Anaonda\lib\site-packages\pandas\core\indexes\datetimes.py in get_loc(self, key, method, tolerance)
        737         except KeyError as err:
    --> 738             raise KeyError(orig_key) from err
        739 
    

    KeyError: '20-01-01'

    
    During handling of the above exception, another exception occurred:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\3719098981.py in <module>
          1 arima = ARIMA(train_data, order=(0, 2 ,0))
          2 results = arima.fit()
    ----> 3 predictions = results.predict('20-01-01', '2014-01-01')
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\base\wrapper.py in wrapper(self, *args, **kwargs)
        111             obj = data.wrap_output(func(results, *args, **kwargs), how[0], how[1:])
        112         elif how:
    --> 113             obj = data.wrap_output(func(results, *args, **kwargs), how)
        114         return obj
        115 
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\mlemodel.py in predict(self, start, end, dynamic, **kwargs)
       3401         """
       3402         # Perform the prediction
    -> 3403         prediction_results = self.get_prediction(start, end, dynamic, **kwargs)
       3404         return prediction_results.predicted_mean
       3405 
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\statespace\mlemodel.py in get_prediction(self, start, end, dynamic, index, exog, extend_model, extend_kwargs, **kwargs)
       3285         # Handle start, end, dynamic
       3286         start, end, out_of_sample, prediction_index = (
    -> 3287             self.model._get_prediction_index(start, end, index))
       3288 
       3289         # Handle `dynamic`
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in _get_prediction_index(self, start, end, index, silent)
        832         """
        833         nobs = len(self.endog)
    --> 834         return get_prediction_index(
        835             start,
        836             end,
    

    ~\AppData\Roaming\Python\Python39\site-packages\statsmodels\tsa\base\tsa_model.py in get_prediction_index(start, end, nobs, base_index, index, silent, index_none, index_generated, data)
        358         )
        359     except KeyError:
    --> 360         raise KeyError(
        361             "The `start` argument could not be matched to a"
        362             " location related to the index of the data."
    

    KeyError: 'The `start` argument could not be matched to a location related to the index of the data.'



```python
from statsmodels.tsa.arima.model import ARIMA

arima2 = ARIMA(df2.loc[:,'First Difference'], order=(15, 1, 2))
results2 = arima2.fit()
predictions2 = results2.predict('2021-01-01', '2021-12-30')

plt.plot(df2.loc['2021-01-01':,'First Difference'],label='Actual')
predictions2.plot()
plt.legend();

print('RMSE:',np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'First Difference'],predictions2)))
print('MAE',mean_absolute_error(df2.loc['2021-01-01':,'First Difference'],predictions2))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\971848261.py in <module>
          9 plt.legend();
         10 
    ---> 11 print('RMSE:',np.sqrt(mean_squared_error(df2.loc['2021-01-01':,'First Difference'],predictions2)))
         12 print('MAE',mean_absolute_error(df2.loc['2021-01-01':,'First Difference'],predictions2))
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
        436     0.825...
        437     """
    --> 438     y_type, y_true, y_pred, multioutput = _check_reg_targets(
        439         y_true, y_pred, multioutput
        440     )
    

    D:\Anaonda\lib\site-packages\sklearn\metrics\_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
         92         the dtype argument passed to check_array.
         93     """
    ---> 94     check_consistent_length(y_true, y_pred)
         95     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
         96     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    

    D:\Anaonda\lib\site-packages\sklearn\utils\validation.py in check_consistent_length(*arrays)
        330     uniques = np.unique(lengths)
        331     if len(uniques) > 1:
    --> 332         raise ValueError(
        333             "Found input variables with inconsistent numbers of samples: %r"
        334             % [int(l) for l in lengths]
    

    ValueError: Found input variables with inconsistent numbers of samples: [359, 358]



    
![png](output_162_1.png)
    



```python

```


```python

```


```python
NEural network
```


```python
train_norm = train_data

#converted into array as all the methods available are for arrays and not lists
train_norm_arr = np.asarray(train_norm)
train_norm = np.reshape(train_norm_arr, (-1, 1))

#Scaling all values between 0 and 1 so that large values don't just dominate
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train_norm)
for i in range(5):
    print(train_norm[i])
```

    [0.68340266]
    [0.38659266]
    [0.51273691]
    [0.36927874]
    [0.38659266]
    


```python
test_norm = test_data
test_norm_arr = np.asarray(test_norm)
test_norm = np.reshape(test_norm_arr, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
test_norm = scaler.fit_transform(test_norm)
for i in range(5):
    print(test_norm[i])
```

    [0.76456632]
    [0.17137867]
    [0.84216737]
    [0.27265461]
    [0.76982741]
    


```python
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X),array(y)
```


```python
n_steps = 3
X_split_train, y_split_train = split_sequence(train_norm, n_steps)
#for i in range(len(X_split_train)):
    #print(X_split_train[i], y_split_train[i])
n_features = 1
X_split_train = X_split_train.reshape((X_split_train.shape[0], X_split_train.shape[1], n_features))
for i in range(5):
    print(X_split_train)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7840\1453386804.py in <module>
          1 n_steps = 3
    ----> 2 X_split_train, y_split_train = split_sequence(train_norm, n_steps)
          3 #for i in range(len(X_split_train)):
          4     #print(X_split_train[i], y_split_train[i])
          5 n_features = 1
    

    ~\AppData\Local\Temp\ipykernel_7840\1392342224.py in split_sequence(sequence, n_steps)
         11         X.append(seq_x)
         12         y.append(seq_y)
    ---> 13     return array(X),array(y)
    

    TypeError: array() argument 1 must be a unicode character, not list



```python

```
