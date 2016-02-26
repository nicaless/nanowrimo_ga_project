
In my first attempt at Logistic Regression I used all the numeric features, but now I want to exclude information from the contest that has already started.  


```python
# import the data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

writers = pd.read_csv("../clean data/user_summary_no2015.csv", index_col=0)
writers.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Writer Name</th>
      <th>Member Length</th>
      <th>LifetimeWordCount</th>
      <th>url</th>
      <th>Age</th>
      <th>Birthday</th>
      <th>Favorite books or authors</th>
      <th>Favorite noveling music</th>
      <th>Hobbies</th>
      <th>Location</th>
      <th>...</th>
      <th>Expected Max Submission</th>
      <th>Expected Max Day</th>
      <th>Expected Std Submissions</th>
      <th>Expected Consec Subs</th>
      <th>FW Total</th>
      <th>FW Sub</th>
      <th>FH Total</th>
      <th>FH Sub</th>
      <th>SH Total</th>
      <th>SH Sub</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nicaless</td>
      <td>2</td>
      <td>50919</td>
      <td>http://nanowrimo.org/participants/nicaless</td>
      <td>24</td>
      <td>December 20</td>
      <td>Ursula Le Guin, J.K.</td>
      <td>Classical, Musicals</td>
      <td>Reading, Video Games, Blogging, Learning</td>
      <td>San Francisco, CA</td>
      <td>...</td>
      <td>24935.0</td>
      <td>28.000000</td>
      <td>6235.712933</td>
      <td>12.000000</td>
      <td>6689</td>
      <td>6</td>
      <td>12486</td>
      <td>9</td>
      <td>11743</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rachel B. Moore</td>
      <td>10</td>
      <td>478090</td>
      <td>http://nanowrimo.org/participants/rachel-b-moore</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2666, Unaccustomed Earth, Exit Music, Crazy Lo...</td>
      <td>Belle and Sebastian, Elliott Smith, PJ Harvey,...</td>
      <td>Reading, volunteering, knitting, listening to ...</td>
      <td>San Francisco</td>
      <td>...</td>
      <td>3809.0</td>
      <td>9.000000</td>
      <td>1002.295167</td>
      <td>6.800000</td>
      <td>16722</td>
      <td>7</td>
      <td>24086</td>
      <td>14</td>
      <td>26517</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abookishbabe</td>
      <td>1</td>
      <td>0</td>
      <td>http://nanowrimo.org/participants/abookishbabe</td>
      <td>NaN</td>
      <td>April 2</td>
      <td>Colleen Hoover, Veronica Roth, Jennifer Niven,...</td>
      <td>Tori Kelley</td>
      <td>Reading (DUH), Day dreaming, Going to Disneyla...</td>
      <td>Sacramento, CA</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28632</td>
      <td>1</td>
      <td>29299</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alexabexis</td>
      <td>11</td>
      <td>475500</td>
      <td>http://nanowrimo.org/participants/alexabexis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Three Goddesses playlist Florence + the Machin...</td>
      <td>drawing, reading, movies &amp; TV shows, comics, p...</td>
      <td>New York City</td>
      <td>...</td>
      <td>2325.0</td>
      <td>8.545455</td>
      <td>570.626795</td>
      <td>8.090909</td>
      <td>25360</td>
      <td>7</td>
      <td>38034</td>
      <td>12</td>
      <td>40766</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AllYellowFlowers</td>
      <td>3</td>
      <td>30428</td>
      <td>http://nanowrimo.org/participants/AllYellowFlo...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Lolita, Jesus' Son, Ask the</td>
      <td>the sound of the coffeemaker</td>
      <td>cryptozoology</td>
      <td>Allston</td>
      <td>...</td>
      <td>2054.5</td>
      <td>4.500000</td>
      <td>538.273315</td>
      <td>21.000000</td>
      <td>1800</td>
      <td>5</td>
      <td>5300</td>
      <td>10</td>
      <td>5700</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>




```python
writers.columns
```




    Index([u'Writer Name', u'Member Length', u'LifetimeWordCount', u'url', u'Age',
           u'Birthday', u'Favorite books or authors', u'Favorite noveling music',
           u'Hobbies', u'Location', u'Occupation', u'Primary Role',
           u'Sponsorship URL', u'Expected Final Word Count',
           u'Expected Daily Average', u'CURRENT WINNER', u'Current Donor', u'Wins',
           u'Donations', u'Participated', u'Consecutive Donor',
           u'Consecutive Wins', u'Consecutive Part', u'Part Years', u'Win Years',
           u'Donor Years', u'Num Novels', u'Expected Num Submissions',
           u'Expected Avg Submission', u'Expected Min Submission',
           u'Expected Min Day', u'Expected Max Submission', u'Expected Max Day',
           u'Expected Std Submissions', u'Expected Consec Subs', u'FW Total',
           u'FW Sub', u'FH Total', u'FH Sub', u'SH Total', u'SH Sub'],
          dtype='object')




```python
# convert primary role and sponsorship url to binary vars
writers['Primary Role'][writers['Primary Role'] == 'Municipal Liaison'] = 1
writers['Primary Role'][writers['Primary Role'] != 1] = 0

writers['Sponsorship URL'].fillna(0, inplace=True)
writers['Sponsorship URL'][writers['Sponsorship URL'] != 0] = 1
```


```python
# let's keep ALL NUMERIC COLUMNS except the CURRENT WINNER column which we will use as response
features = writers._get_numeric_data()
features.columns
```




    Index([u'Member Length', u'LifetimeWordCount', u'Age',
           u'Expected Final Word Count', u'Expected Daily Average',
           u'CURRENT WINNER', u'Current Donor', u'Wins', u'Donations',
           u'Participated', u'Consecutive Donor', u'Consecutive Wins',
           u'Consecutive Part', u'Num Novels', u'Expected Num Submissions',
           u'Expected Avg Submission', u'Expected Min Submission',
           u'Expected Min Day', u'Expected Max Submission', u'Expected Max Day',
           u'Expected Std Submissions', u'Expected Consec Subs', u'FW Total',
           u'FW Sub', u'FH Total', u'FH Sub', u'SH Total', u'SH Sub'],
          dtype='object')




```python
del features['CURRENT WINNER']
# delete features that would only be collected after a contest starts
#del features['Current Donor']
del features['FW Total']
del features['FW Sub']
del features['FH Total']
del features['FH Sub']
del features['SH Total']
del features['SH Sub']


features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member Length</th>
      <th>LifetimeWordCount</th>
      <th>Age</th>
      <th>Expected Final Word Count</th>
      <th>Expected Daily Average</th>
      <th>Current Donor</th>
      <th>Wins</th>
      <th>Donations</th>
      <th>Participated</th>
      <th>Consecutive Donor</th>
      <th>...</th>
      <th>Consecutive Part</th>
      <th>Num Novels</th>
      <th>Expected Num Submissions</th>
      <th>Expected Avg Submission</th>
      <th>Expected Min Submission</th>
      <th>Expected Min Day</th>
      <th>Expected Max Submission</th>
      <th>Expected Max Day</th>
      <th>Expected Std Submissions</th>
      <th>Expected Consec Subs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>50919</td>
      <td>24</td>
      <td>50919.000000</td>
      <td>1697.300000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>14.000000</td>
      <td>3637.071429</td>
      <td>299.0</td>
      <td>2.000000</td>
      <td>24935.0</td>
      <td>28.000000</td>
      <td>6235.712933</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>478090</td>
      <td>NaN</td>
      <td>47809.000000</td>
      <td>1593.633333</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>10</td>
      <td>8</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>8.300000</td>
      <td>918.057453</td>
      <td>42.7</td>
      <td>7.700000</td>
      <td>3809.0</td>
      <td>9.000000</td>
      <td>1002.295167</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>475500</td>
      <td>NaN</td>
      <td>43227.272727</td>
      <td>1440.909091</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>11</td>
      <td>4</td>
      <td>...</td>
      <td>11</td>
      <td>11</td>
      <td>9.272727</td>
      <td>822.780595</td>
      <td>36.0</td>
      <td>6.727273</td>
      <td>2325.0</td>
      <td>8.545455</td>
      <td>570.626795</td>
      <td>8.090909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>30428</td>
      <td>NaN</td>
      <td>15214.000000</td>
      <td>507.133333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>22.000000</td>
      <td>678.318083</td>
      <td>50.0</td>
      <td>10.500000</td>
      <td>2054.5</td>
      <td>4.500000</td>
      <td>538.273315</td>
      <td>21.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
y = writers['CURRENT WINNER'].values
```


```python
# inputting 0 for users without prior data for daily avg, avg submission, num submissions etc. and so are marked NaN
features.fillna(0, inplace=True)
features.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member Length</th>
      <th>LifetimeWordCount</th>
      <th>Age</th>
      <th>Expected Final Word Count</th>
      <th>Expected Daily Average</th>
      <th>Current Donor</th>
      <th>Wins</th>
      <th>Donations</th>
      <th>Participated</th>
      <th>Consecutive Donor</th>
      <th>...</th>
      <th>Consecutive Part</th>
      <th>Num Novels</th>
      <th>Expected Num Submissions</th>
      <th>Expected Avg Submission</th>
      <th>Expected Min Submission</th>
      <th>Expected Min Day</th>
      <th>Expected Max Submission</th>
      <th>Expected Max Day</th>
      <th>Expected Std Submissions</th>
      <th>Expected Consec Subs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>...</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.212575</td>
      <td>172552.676647</td>
      <td>8.596806</td>
      <td>36428.312194</td>
      <td>1214.277073</td>
      <td>0.317365</td>
      <td>2.606786</td>
      <td>1.421158</td>
      <td>3.656687</td>
      <td>1.047904</td>
      <td>...</td>
      <td>3.057884</td>
      <td>3.377246</td>
      <td>10.826177</td>
      <td>1708.026777</td>
      <td>73.105821</td>
      <td>6.128300</td>
      <td>4764.389341</td>
      <td>10.005534</td>
      <td>1314.411102</td>
      <td>9.573348</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.255209</td>
      <td>329113.331830</td>
      <td>14.463648</td>
      <td>43782.218313</td>
      <td>1459.407277</td>
      <td>0.465916</td>
      <td>4.651782</td>
      <td>3.044384</td>
      <td>4.899582</td>
      <td>1.760029</td>
      <td>...</td>
      <td>2.946632</td>
      <td>3.451290</td>
      <td>8.520344</td>
      <td>2053.622361</td>
      <td>1566.761571</td>
      <td>6.145692</td>
      <td>5727.358954</td>
      <td>8.406292</td>
      <td>2011.241171</td>
      <td>8.393503</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-21113.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>9818.000000</td>
      <td>0.000000</td>
      <td>7443.250000</td>
      <td>248.108333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>362.750000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>955.000000</td>
      <td>1.000000</td>
      <td>256.685927</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>93385.000000</td>
      <td>0.000000</td>
      <td>37594.333333</td>
      <td>1253.144444</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>10.250000</td>
      <td>1446.652778</td>
      <td>85.666667</td>
      <td>4.500000</td>
      <td>3546.500000</td>
      <td>9.333333</td>
      <td>873.018486</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>206482.000000</td>
      <td>20.000000</td>
      <td>50734.200000</td>
      <td>1691.140000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>17.333333</td>
      <td>2213.520000</td>
      <td>291.500000</td>
      <td>10.000000</td>
      <td>6250.000000</td>
      <td>16.200000</td>
      <td>1516.145753</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.000000</td>
      <td>4562712.000000</td>
      <td>61.000000</td>
      <td>651816.000000</td>
      <td>21727.200000</td>
      <td>1.000000</td>
      <td>52.000000</td>
      <td>36.000000</td>
      <td>52.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>14.000000</td>
      <td>26.000000</td>
      <td>30.000000</td>
      <td>20869.236584</td>
      <td>5000.000000</td>
      <td>27.666667</td>
      <td>51238.000000</td>
      <td>30.000000</td>
      <td>23874.872328</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d
output_notebook()
```




    <script type="text/javascript">
      
      (function(global) {
        function now() {
          return new Date();
        }
      
        if (typeof (window._bokeh_onload_callbacks) === "undefined") {
          window._bokeh_onload_callbacks = [];
        }
      
        function run_callbacks() {
          window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
          delete window._bokeh_onload_callbacks
          console.info("Bokeh: all callbacks have finished");
        }
      
        function load_libs(js_urls, callback) {
          window._bokeh_onload_callbacks.push(callback);
          if (window._bokeh_is_loading > 0) {
            console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
            return null;
          }
          if (js_urls == null || js_urls.length === 0) {
            run_callbacks();
            return null;
          }
          console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
          window._bokeh_is_loading = js_urls.length;
          for (var i = 0; i < js_urls.length; i++) {
            var url = js_urls[i];
            var s = document.createElement('script');
            s.src = url;
            s.async = false;
            s.onreadystatechange = s.onload = function() {
              window._bokeh_is_loading--;
              if (window._bokeh_is_loading === 0) {
                console.log("Bokeh: all BokehJS libraries loaded");
                run_callbacks()
              }
            };
            s.onerror = function() {
              console.warn("failed to load library " + url);
            };
            console.log("Bokeh: injecting script tag for BokehJS library: ", url);
            document.getElementsByTagName("head")[0].appendChild(s);
          }
        };var js_urls = ['https://cdn.pydata.org/bokeh/release/bokeh-0.11.0.min.js', 'https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.11.0.min.js', 'https://cdn.pydata.org/bokeh/release/bokeh-compiler-0.11.0.min.js'];
      
        var inline_js = [
          function(Bokeh) {
            Bokeh.set_log_level("info");
          },
          function(Bokeh) {
            console.log("Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-0.11.0.min.css");
            Bokeh.embed.inject_css("https://cdn.pydata.org/bokeh/release/bokeh-0.11.0.min.css");
            console.log("Bokeh: injecting CSS: https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.11.0.min.css");
            Bokeh.embed.inject_css("https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.11.0.min.css");
          }
        ];
      
        function run_inline_js() {
          for (var i = 0; i < inline_js.length; i++) {
            inline_js[i](window.Bokeh);
          }
        }
      
        if (window._bokeh_is_loading === 0) {
          console.log("Bokeh: BokehJS loaded, going straight to plotting");
          run_inline_js();
        } else {
          load_libs(js_urls, function() {
            console.log("Bokeh: BokehJS plotting callback run at", now());
            run_inline_js();
          });
        }
      }(this));
    </script>
    <div>
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span>BokehJS successfully loaded.</span>
    </div>


### Normalize data


```python
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)
features_norm[1]
```




    array([ 1.77967343,  0.92929298, -0.5949674 ,  0.26019839,  0.26019839,
            1.46660949,  1.16054543,  2.1631367 ,  1.29595831,  3.95393831,
            1.98638802,  2.35830378,  1.92083955, -0.29678396, -0.38505565,
           -0.01942619,  0.25599572, -0.16697823, -0.11973645, -0.15534084,
           -0.33074635])



### Apply Logistic Regression


```python
def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
    
    p = figure(title='Receiver Operating Characteristic')
    # Plot ROC curve
    p.line(x=fpr,y=tpr,legend='ROC curve (area = %0.3f)' % roc_auc)
    p.x_range=Range1d(0,1)
    p.y_range=Range1d(0,1)
    p.xaxis.axis_label='False Positive Rate or (1 - Specifity)'
    p.yaxis.axis_label='True Positive Rate or (Sensitivity)'
    p.legend.location = "bottom_right"
    show(p)
    
%matplotlib inline
```


```python
X_train, X_test, y_train, y_test = train_test_split(features_norm,y, test_size=0.2, random_state=0)
```


```python
model_lr = LogisticRegression(C=5)
print cross_val_score(model_lr,X_train, y_train,cv=10).mean()

model_lr.fit(X_train, y_train)
print pd.DataFrame(confusion_matrix(y_test,model_lr.predict(X_test)), index=['Predicted Class 0', 'Predicted Class 1'], 
                     columns=['Actual Class 0', 'Actual Class 1'])
print classification_report(y_test,model_lr.predict(X_test))
print model_lr.score(X_test,y_test)
plot_roc_curve(y_test, model_lr.predict_proba(X_test))

```

    0.68538148843
                       Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              22              24
                 precision    recall  f1-score   support
    
              0       0.69      0.87      0.77        55
              1       0.77      0.52      0.62        46
    
    avg / total       0.73      0.71      0.70       101
    
    0.712871287129





    <div class="plotdiv" id="c4ed1878-1aca-4422-853d-8b49e2344397"></div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined") {
      window._bokeh_onload_callbacks = [];
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("c4ed1878-1aca-4422-853d-8b49e2344397");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'c4ed1878-1aca-4422-853d-8b49e2344397' but no matching script tag was found. ")
      return false;
    }var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        Bokeh.$(function() {
            var docs_json = {"4e6df88c-c488-47ab-b8af-be1eb55cb6f4": {"version": "0.11.0", "roots": {"root_ids": ["d8c0df46-779b-473c-97c3-a744eb488076", "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"], "references": [{"attributes": {"callback": null}, "type": "Range1d", "id": "bb9cb5d4-f238-4c59-8d1f-a679198f9901"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "line_alpha": {"value": 0.1}, "y": {"field": "y"}, "x": {"field": "x"}}, "type": "Line", "id": "f45bc650-30ce-4c28-b142-363325b13595"}, {"attributes": {"callback": null}, "type": "Range1d", "id": "7fc38608-4466-4e3e-917e-518f66173a18"}, {"attributes": {"callback": null, "column_names": ["y", "x"], "data": {"y": [0.021739130434782608, 0.13043478260869565, 0.13043478260869565, 0.391304347826087, 0.391304347826087, 0.41304347826086957, 0.41304347826086957, 0.43478260869565216, 0.43478260869565216, 0.45652173913043476, 0.45652173913043476, 0.5652173913043478, 0.5652173913043478, 0.6086956521739131, 0.6086956521739131, 0.6739130434782609, 0.6739130434782609, 0.6956521739130435, 0.7391304347826086, 0.7608695652173914, 0.7608695652173914, 0.7608695652173914, 0.7608695652173914, 0.8043478260869565, 0.8043478260869565, 0.8260869565217391, 0.8260869565217391, 0.8478260869565217, 0.8478260869565217, 0.8695652173913043, 0.8695652173913043, 0.8913043478260869, 0.8913043478260869, 0.9130434782608695, 0.9130434782608695, 0.9347826086956522, 0.9347826086956522, 0.9565217391304348, 0.9565217391304348, 0.9782608695652174, 0.9782608695652174, 1.0, 1.0], "x": [0.0, 0.0, 0.01818181818181818, 0.01818181818181818, 0.05454545454545454, 0.05454545454545454, 0.07272727272727272, 0.07272727272727272, 0.10909090909090909, 0.10909090909090909, 0.12727272727272726, 0.12727272727272726, 0.21818181818181817, 0.21818181818181817, 0.23636363636363636, 0.23636363636363636, 0.3090909090909091, 0.3090909090909091, 0.3090909090909091, 0.32727272727272727, 0.34545454545454546, 0.4, 0.43636363636363634, 0.43636363636363634, 0.45454545454545453, 0.45454545454545453, 0.4727272727272727, 0.4727272727272727, 0.5272727272727272, 0.5272727272727272, 0.6, 0.6, 0.6181818181818182, 0.6181818181818182, 0.6363636363636364, 0.6363636363636364, 0.6545454545454545, 0.6545454545454545, 0.7818181818181819, 0.7818181818181819, 0.9818181818181818, 0.9818181818181818, 1.0]}}, "type": "ColumnDataSource", "id": "046c1d27-4d50-435f-b095-15d8d63771e8"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "ticker": {"type": "BasicTicker", "id": "294081d1-8dd4-46e0-80b2-e2f86ed876e2"}, "dimension": 1}, "type": "Grid", "id": "75249294-206a-4f45-abfa-b90ac2505988"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "PanTool", "id": "45f1d66c-e978-4600-a197-67f75ed92895"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "WheelZoomTool", "id": "6e84d1f0-3d22-4f15-bc9c-e97a4f24a8fc"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "PanTool", "id": "5cd1c3f8-dded-46f4-ab01-224ad04a16ac"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "ResetTool", "id": "44f23611-1ac6-46c0-9c0c-f467512ae0c0"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "axis_label": "True Positive Rate or (Sensitivity)", "formatter": {"type": "BasicTickFormatter", "id": "5bc0098d-b2de-4fd6-bb07-3b1fd28f4a54"}, "ticker": {"type": "BasicTicker", "id": "cf822081-fb00-40ab-80ae-7d9a35f4267f"}}, "type": "LinearAxis", "id": "8c9ec8b7-59b5-41d0-a5ef-c2d334490aae"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "overlay": {"type": "BoxAnnotation", "id": "57f0aa27-e8d4-4b19-8c7c-c595b93084df"}}, "type": "BoxZoomTool", "id": "6684c4d0-a3a7-42e0-b6e4-bf7cb578eb5d"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "ticker": {"type": "BasicTicker", "id": "8bea6c50-6b9a-42f0-b22d-470228176d17"}}, "type": "Grid", "id": "d29adc3d-66b0-4eeb-8d43-7456399a3eb8"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "axis_label": "False Positive Rate or (1 - Specifity)", "formatter": {"type": "BasicTickFormatter", "id": "e1182439-2cdf-4373-b615-52797b0911a4"}, "ticker": {"type": "BasicTicker", "id": "cc5c47fc-a508-4eb6-9911-7a84f8fe9971"}}, "type": "LinearAxis", "id": "ce6370bb-cbc4-40c0-a63d-84cdae7a22a6"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "ResizeTool", "id": "fe1bb868-dcce-4199-884e-dea6b427f8f7"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "HelpTool", "id": "dfb85140-04d4-4869-a18a-bafe2051f959"}, {"attributes": {"nonselection_glyph": {"type": "Line", "id": "6486d07d-777c-4da7-b40e-fca637a0e3a4"}, "data_source": {"type": "ColumnDataSource", "id": "046c1d27-4d50-435f-b095-15d8d63771e8"}, "selection_glyph": null, "hover_glyph": null, "glyph": {"type": "Line", "id": "508bcf54-1d3e-4b4f-8f3a-6c187b5b9f5c"}}, "type": "GlyphRenderer", "id": "aa6e9e26-2897-48f7-b5f4-4107aa5ab16d"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "HelpTool", "id": "c5a7e089-099a-44d1-b675-c6ad7146df29"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "PreviewSaveTool", "id": "d418256d-6ce5-42a4-99ac-41e072335943"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "ticker": {"type": "BasicTicker", "id": "cc5c47fc-a508-4eb6-9911-7a84f8fe9971"}}, "type": "Grid", "id": "9a54515e-d916-403c-a592-f3a7d5beed3d"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "ticker": {"type": "BasicTicker", "id": "cf822081-fb00-40ab-80ae-7d9a35f4267f"}, "dimension": 1}, "type": "Grid", "id": "a93563f5-97af-4d89-87d3-ed2dec2d4cbe"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "ResetTool", "id": "359ed043-1480-42dd-8f8c-257b50ee3f9c"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "axis_label": "False Positive Rate or (1 - Specifity)", "formatter": {"type": "BasicTickFormatter", "id": "0a7901cd-d39c-47cd-af98-7a0e401f61af"}, "ticker": {"type": "BasicTicker", "id": "8bea6c50-6b9a-42f0-b22d-470228176d17"}}, "type": "LinearAxis", "id": "6d8349d5-ad64-4901-8eb5-717cf60aef4a"}, {"attributes": {"callback": null}, "type": "Range1d", "id": "d58406d3-af8e-438b-a8a7-9feadcad5648"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "axis_label": "True Positive Rate or (Sensitivity)", "formatter": {"type": "BasicTickFormatter", "id": "095d0331-7f2c-4636-bf40-ef0255dcaaf2"}, "ticker": {"type": "BasicTicker", "id": "294081d1-8dd4-46e0-80b2-e2f86ed876e2"}}, "type": "LinearAxis", "id": "25b923ab-9ea0-4630-8701-2e0a69c89ec4"}, {"attributes": {"line_color": {"value": "black"}, "line_alpha": {"value": 1.0}, "render_mode": "css", "bottom_units": "screen", "level": "overlay", "top_units": "screen", "fill_alpha": {"value": 0.5}, "plot": null, "left_units": "screen", "line_dash": [4, 4], "line_width": {"value": 2}, "right_units": "screen", "fill_color": {"value": "lightgrey"}}, "type": "BoxAnnotation", "id": "f4e81689-235e-465b-a6f4-a12dace80dc5"}, {"attributes": {}, "type": "BasicTickFormatter", "id": "0a7901cd-d39c-47cd-af98-7a0e401f61af"}, {"attributes": {"x_range": {"type": "Range1d", "id": "203058ef-2597-488d-84ae-96403e209dbc"}, "title": "Receiver Operating Characteristic", "y_range": {"type": "Range1d", "id": "d58406d3-af8e-438b-a8a7-9feadcad5648"}, "renderers": [{"type": "LinearAxis", "id": "ce6370bb-cbc4-40c0-a63d-84cdae7a22a6"}, {"type": "Grid", "id": "9a54515e-d916-403c-a592-f3a7d5beed3d"}, {"type": "LinearAxis", "id": "25b923ab-9ea0-4630-8701-2e0a69c89ec4"}, {"type": "Grid", "id": "75249294-206a-4f45-abfa-b90ac2505988"}, {"type": "BoxAnnotation", "id": "f4e81689-235e-465b-a6f4-a12dace80dc5"}, {"type": "Legend", "id": "2efa5871-639f-4c28-b3c4-f491f6b4b2e4"}, {"type": "GlyphRenderer", "id": "f73d4e19-80e9-4fa1-966d-f91e15168c69"}], "below": [{"type": "LinearAxis", "id": "ce6370bb-cbc4-40c0-a63d-84cdae7a22a6"}], "tool_events": {"type": "ToolEvents", "id": "1c287721-f2d9-43fa-a518-92c0bce02899"}, "tools": [{"type": "PanTool", "id": "45f1d66c-e978-4600-a197-67f75ed92895"}, {"type": "WheelZoomTool", "id": "6e84d1f0-3d22-4f15-bc9c-e97a4f24a8fc"}, {"type": "BoxZoomTool", "id": "8ce52130-8a66-4060-bcae-21abf470c6f0"}, {"type": "PreviewSaveTool", "id": "d418256d-6ce5-42a4-99ac-41e072335943"}, {"type": "ResizeTool", "id": "dbb7dca3-fd1d-4755-be8d-a10bfee4d616"}, {"type": "ResetTool", "id": "359ed043-1480-42dd-8f8c-257b50ee3f9c"}, {"type": "HelpTool", "id": "c5a7e089-099a-44d1-b675-c6ad7146df29"}], "left": [{"type": "LinearAxis", "id": "25b923ab-9ea0-4630-8701-2e0a69c89ec4"}]}, "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076", "subtype": "Figure"}, {"attributes": {"nonselection_glyph": {"type": "Line", "id": "f45bc650-30ce-4c28-b142-363325b13595"}, "data_source": {"type": "ColumnDataSource", "id": "911d955f-4b6a-4bf9-8c33-482c6e5b0c50"}, "selection_glyph": null, "hover_glyph": null, "glyph": {"type": "Line", "id": "a470cc67-2d5b-4958-b5c9-a05379a03614"}}, "type": "GlyphRenderer", "id": "f73d4e19-80e9-4fa1-966d-f91e15168c69"}, {"attributes": {}, "type": "BasicTickFormatter", "id": "5bc0098d-b2de-4fd6-bb07-3b1fd28f4a54"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "overlay": {"type": "BoxAnnotation", "id": "f4e81689-235e-465b-a6f4-a12dace80dc5"}}, "type": "BoxZoomTool", "id": "8ce52130-8a66-4060-bcae-21abf470c6f0"}, {"attributes": {}, "type": "BasicTicker", "id": "294081d1-8dd4-46e0-80b2-e2f86ed876e2"}, {"attributes": {}, "type": "BasicTicker", "id": "8bea6c50-6b9a-42f0-b22d-470228176d17"}, {"attributes": {}, "type": "BasicTickFormatter", "id": "e1182439-2cdf-4373-b615-52797b0911a4"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "PreviewSaveTool", "id": "bec2d03f-2aab-4f33-a32e-b26df1c2b361"}, {"attributes": {}, "type": "BasicTicker", "id": "cf822081-fb00-40ab-80ae-7d9a35f4267f"}, {"attributes": {"line_color": {"value": "black"}, "line_alpha": {"value": 1.0}, "render_mode": "css", "bottom_units": "screen", "level": "overlay", "top_units": "screen", "fill_alpha": {"value": 0.5}, "plot": null, "left_units": "screen", "line_dash": [4, 4], "line_width": {"value": 2}, "right_units": "screen", "fill_color": {"value": "lightgrey"}}, "type": "BoxAnnotation", "id": "57f0aa27-e8d4-4b19-8c7c-c595b93084df"}, {"attributes": {"callback": null}, "type": "Range1d", "id": "203058ef-2597-488d-84ae-96403e209dbc"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}}, "type": "WheelZoomTool", "id": "b9c8c12c-3b29-45b9-81e5-7945c11d10ee"}, {"attributes": {"x_range": {"type": "Range1d", "id": "7fc38608-4466-4e3e-917e-518f66173a18"}, "title": "Receiver Operating Characteristic", "y_range": {"type": "Range1d", "id": "bb9cb5d4-f238-4c59-8d1f-a679198f9901"}, "renderers": [{"type": "LinearAxis", "id": "6d8349d5-ad64-4901-8eb5-717cf60aef4a"}, {"type": "Grid", "id": "d29adc3d-66b0-4eeb-8d43-7456399a3eb8"}, {"type": "LinearAxis", "id": "8c9ec8b7-59b5-41d0-a5ef-c2d334490aae"}, {"type": "Grid", "id": "a93563f5-97af-4d89-87d3-ed2dec2d4cbe"}, {"type": "BoxAnnotation", "id": "57f0aa27-e8d4-4b19-8c7c-c595b93084df"}, {"type": "Legend", "id": "48c25f9a-ff32-4118-9d1f-d9b7efbb7a54"}, {"type": "GlyphRenderer", "id": "aa6e9e26-2897-48f7-b5f4-4107aa5ab16d"}], "below": [{"type": "LinearAxis", "id": "6d8349d5-ad64-4901-8eb5-717cf60aef4a"}], "tool_events": {"type": "ToolEvents", "id": "70fdb057-5cd3-43f2-9c88-c17197a4d872"}, "tools": [{"type": "PanTool", "id": "5cd1c3f8-dded-46f4-ab01-224ad04a16ac"}, {"type": "WheelZoomTool", "id": "b9c8c12c-3b29-45b9-81e5-7945c11d10ee"}, {"type": "BoxZoomTool", "id": "6684c4d0-a3a7-42e0-b6e4-bf7cb578eb5d"}, {"type": "PreviewSaveTool", "id": "bec2d03f-2aab-4f33-a32e-b26df1c2b361"}, {"type": "ResizeTool", "id": "fe1bb868-dcce-4199-884e-dea6b427f8f7"}, {"type": "ResetTool", "id": "44f23611-1ac6-46c0-9c0c-f467512ae0c0"}, {"type": "HelpTool", "id": "dfb85140-04d4-4869-a18a-bafe2051f959"}], "left": [{"type": "LinearAxis", "id": "8c9ec8b7-59b5-41d0-a5ef-c2d334490aae"}]}, "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5", "subtype": "Figure"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "x": {"field": "x"}, "y": {"field": "y"}}, "type": "Line", "id": "a470cc67-2d5b-4958-b5c9-a05379a03614"}, {"attributes": {"callback": null, "column_names": ["y", "x"], "data": {"y": [0.021739130434782608, 0.17391304347826086, 0.17391304347826086, 0.2826086956521739, 0.2826086956521739, 0.32608695652173914, 0.32608695652173914, 0.34782608695652173, 0.34782608695652173, 0.45652173913043476, 0.45652173913043476, 0.5, 0.5, 0.5434782608695652, 0.5434782608695652, 0.6086956521739131, 0.6086956521739131, 0.6304347826086957, 0.6304347826086957, 0.6521739130434783, 0.6956521739130435, 0.717391304347826, 0.717391304347826, 0.7608695652173914, 0.7608695652173914, 0.7608695652173914, 0.8260869565217391, 0.8260869565217391, 0.8695652173913043, 0.8695652173913043, 0.9130434782608695, 0.9130434782608695, 0.9347826086956522, 0.9347826086956522, 0.9565217391304348, 0.9565217391304348, 0.9782608695652174, 0.9782608695652174, 1.0, 1.0], "x": [0.0, 0.0, 0.01818181818181818, 0.01818181818181818, 0.03636363636363636, 0.03636363636363636, 0.05454545454545454, 0.05454545454545454, 0.09090909090909091, 0.09090909090909091, 0.10909090909090909, 0.10909090909090909, 0.16363636363636364, 0.16363636363636364, 0.18181818181818182, 0.18181818181818182, 0.23636363636363636, 0.23636363636363636, 0.2545454545454545, 0.2545454545454545, 0.2545454545454545, 0.2545454545454545, 0.2727272727272727, 0.3090909090909091, 0.36363636363636365, 0.43636363636363634, 0.43636363636363634, 0.4909090909090909, 0.4909090909090909, 0.6181818181818182, 0.6181818181818182, 0.6545454545454545, 0.6545454545454545, 0.6727272727272727, 0.6727272727272727, 0.7818181818181819, 0.7818181818181819, 0.8909090909090909, 0.8909090909090909, 1.0]}}, "type": "ColumnDataSource", "id": "911d955f-4b6a-4bf9-8c33-482c6e5b0c50"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "x": {"field": "x"}, "y": {"field": "y"}}, "type": "Line", "id": "508bcf54-1d3e-4b4f-8f3a-6c187b5b9f5c"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}, "legends": [["ROC curve (area = 0.780)", [{"type": "GlyphRenderer", "id": "f73d4e19-80e9-4fa1-966d-f91e15168c69"}]]], "location": "bottom_right"}, "type": "Legend", "id": "2efa5871-639f-4c28-b3c4-f491f6b4b2e4"}, {"attributes": {}, "type": "ToolEvents", "id": "1c287721-f2d9-43fa-a518-92c0bce02899"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}, "legends": [["ROC curve (area = 0.781)", [{"type": "GlyphRenderer", "id": "aa6e9e26-2897-48f7-b5f4-4107aa5ab16d"}]]], "location": "bottom_right"}, "type": "Legend", "id": "48c25f9a-ff32-4118-9d1f-d9b7efbb7a54"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "line_alpha": {"value": 0.1}, "y": {"field": "y"}, "x": {"field": "x"}}, "type": "Line", "id": "6486d07d-777c-4da7-b40e-fca637a0e3a4"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "d8c0df46-779b-473c-97c3-a744eb488076"}}, "type": "ResizeTool", "id": "dbb7dca3-fd1d-4755-be8d-a10bfee4d616"}, {"attributes": {}, "type": "BasicTickFormatter", "id": "095d0331-7f2c-4636-bf40-ef0255dcaaf2"}, {"attributes": {}, "type": "BasicTicker", "id": "cc5c47fc-a508-4eb6-9911-7a84f8fe9971"}, {"attributes": {}, "type": "ToolEvents", "id": "70fdb057-5cd3-43f2-9c88-c17197a4d872"}]}, "title": "Bokeh Application"}};
            var render_items = [{"notebook_comms_target": "d53ea6c0-e031-4e2d-977e-fbe1ec9808ce", "docid": "4e6df88c-c488-47ab-b8af-be1eb55cb6f4", "elementid": "c4ed1878-1aca-4422-853d-8b49e2344397", "modelid": "0e974ef5-5fd6-42b7-817d-eae367ed5dc5"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
        });
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      for (var i = 0; i < inline_js.length; i++) {
        inline_js[i](window.Bokeh);
      }
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>


This is not as accurate as when including current contest data.  We can assume then that activity in the first couple weeks of the contest is predictive of winning.  

Still, let's try some other models and see how they do.



### Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
print cross_val_score(model_nb, X_train, y_train, cv=10).mean()

model_nb.fit(X_train, y_train)
print pd.DataFrame(confusion_matrix(y_test,model_nb.predict(X_test)), index=['Predicted Class 0', 'Predicted Class 1'], 
                     columns=['Actual Class 0', 'Actual Class 1'])
print classification_report(y_test,model_nb.predict(X_test))
print model_nb.score(X_test,y_test)
```

    0.665
                       Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              26              20
                 precision    recall  f1-score   support
    
              0       0.65      0.87      0.74        55
              1       0.74      0.43      0.55        46
    
    avg / total       0.69      0.67      0.65       101
    
    0.673267326733


Naive Bayes is not as accurate as Logistic Regression in this case.

### SVM


```python
from sklearn.svm import SVC

model_svc = SVC(kernel="rbf",C=1)
print cross_val_score(model_svc, X_train, y_train, cv=10).mean()

model_svc.fit(X_train, y_train)
print pd.DataFrame(confusion_matrix(y_test,model_svc.predict(X_test)), index=['Predicted Class 0', 'Predicted Class 1'], 
                     columns=['Actual Class 0', 'Actual Class 1'])
print classification_report(y_test,model_svc.predict(X_test))
print model_svc.score(X_test,y_test)
```

    0.703083176986
                       Actual Class 0  Actual Class 1
    Predicted Class 0              49               6
    Predicted Class 1              20              26
                 precision    recall  f1-score   support
    
              0       0.71      0.89      0.79        55
              1       0.81      0.57      0.67        46
    
    avg / total       0.76      0.74      0.73       101
    
    0.742574257426


This Support Vector Machine does a little bit better than the Logistic Regression.

### Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
model_dt = DecisionTreeClassifier(max_depth=5)
print cross_val_score(model_dt, X_train, y_train, cv=10).mean()

model_dt.fit(X_train, y_train)
print pd.DataFrame(confusion_matrix(y_test,model_dt.predict(X_test)), index=['Predicted Class 0', 'Predicted Class 1'], 
                     columns=['Actual Class 0', 'Actual Class 1'])
print classification_report(y_test,model_dt.predict(X_test))
print model_dt.score(X_test,y_test)
```

    0.645051594747
                       Actual Class 0  Actual Class 1
    Predicted Class 0              42              13
    Predicted Class 1              22              24
                 precision    recall  f1-score   support
    
              0       0.66      0.76      0.71        55
              1       0.65      0.52      0.58        46
    
    avg / total       0.65      0.65      0.65       101
    
    0.653465346535


The Decision Tree did not do as well this time without the other features.


```python
dt_importances = pd.DataFrame(zip(features.columns, model_dt.feature_importances_))
dt_importances.sort_values(1, ascending=False).head() # most to least predictive of being 0
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Expected Daily Average</td>
      <td>0.313585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LifetimeWordCount</td>
      <td>0.172179</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Expected Num Submissions</td>
      <td>0.078782</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Expected Min Submission</td>
      <td>0.076582</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Member Length</td>
      <td>0.070148</td>
    </tr>
  </tbody>
</table>
</div>



Without the data from the current contest, the most important features are Expected Daily Average and LifetimeWordCount, or a writer's average daily writing productivity and how much they've participated in the past.

### Random Forests


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
model_rf = RandomForestClassifier(max_depth=5, n_estimators=100)
print cross_val_score(model_rf, X_train, y_train, cv=10).mean()

model_rf.fit(X_train, y_train)
print pd.DataFrame(confusion_matrix(y_test,model_rf.predict(X_test)), index=['Predicted Class 0', 'Predicted Class 1'], 
                     columns=['Actual Class 0', 'Actual Class 1'])
print classification_report(y_test,model_rf.predict(X_test))
print model_rf.score(X_test,y_test)
```

    0.687890869293
                       Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              18              28
                 precision    recall  f1-score   support
    
              0       0.73      0.87      0.79        55
              1       0.80      0.61      0.69        46
    
    avg / total       0.76      0.75      0.75       101
    
    0.752475247525


It looks like Random Forests and Support Vector Machines do best in predicting winners and non-winners when excluding data from the current contest.


```python

```
