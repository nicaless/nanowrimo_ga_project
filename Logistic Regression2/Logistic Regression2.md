
# Predicting NaNoWriMo winners with Logistic Regression

As the variable I want to predict is binary (1 if a writer is a winner, 0 if otherwise) I decided to use a logistic regression as my prediction model.  


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
# let's keep ALL NUMERIAL COLUMNS except the CURRENT WINNER column which we will use as response
features = writers._get_numeric_data()
```


```python
del features['CURRENT WINNER']
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
<p>5 rows × 27 columns</p>
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
      <td>4764.389341</td>
      <td>10.005534</td>
      <td>1314.411102</td>
      <td>9.573348</td>
      <td>12203.137725</td>
      <td>4.413174</td>
      <td>20962.403194</td>
      <td>8.137725</td>
      <td>17100.556886</td>
      <td>6.520958</td>
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
      <td>5727.358954</td>
      <td>8.406292</td>
      <td>2011.241171</td>
      <td>8.393503</td>
      <td>39000.987493</td>
      <td>2.614373</td>
      <td>54462.877403</td>
      <td>5.140330</td>
      <td>21562.099582</td>
      <td>6.259238</td>
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
      <td>0.000000</td>
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
      <td>955.000000</td>
      <td>1.000000</td>
      <td>256.685927</td>
      <td>0.000000</td>
      <td>2258.000000</td>
      <td>2.000000</td>
      <td>3925.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
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
      <td>3546.500000</td>
      <td>9.333333</td>
      <td>873.018486</td>
      <td>8.500000</td>
      <td>7890.000000</td>
      <td>5.000000</td>
      <td>15212.000000</td>
      <td>10.000000</td>
      <td>10900.000000</td>
      <td>5.000000</td>
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
      <td>6250.000000</td>
      <td>16.200000</td>
      <td>1516.145753</td>
      <td>16.000000</td>
      <td>12361.000000</td>
      <td>7.000000</td>
      <td>23832.000000</td>
      <td>13.000000</td>
      <td>28005.000000</td>
      <td>13.000000</td>
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
      <td>51238.000000</td>
      <td>30.000000</td>
      <td>23874.872328</td>
      <td>30.000000</td>
      <td>630036.000000</td>
      <td>7.000000</td>
      <td>1000000.000000</td>
      <td>14.000000</td>
      <td>210000.000000</td>
      <td>16.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```

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
           -0.33074635,  0.11598114,  0.99045236,  0.05741009,  1.14158714,
            0.43714921,  1.19607487])



### Apply Logistic Regression


```python
X_train, X_test, y_train, y_test = train_test_split(features_norm,y, test_size=0.2, random_state=0)
```


```python
model_lr = LogisticRegression(C=5)
cross_val_score(model_lr,X_train, y_train,cv=10).mean()
```




    0.97749374609130713



Wow! That's a very good cross validation score! Now let's check out the model's confusion matrix and classification report and how well it does predicting the targets of the test data.




```python
model_lr.fit(X_train,y_train)
print confusion_matrix(y_test,model_lr.predict(X_test))
print classification_report(y_test,model_lr.predict(X_test))
print model_lr.score(X_test,y_test)
```

    [[51  4]
     [ 0 46]]
                 precision    recall  f1-score   support
    
              0       1.00      0.93      0.96        55
              1       0.92      1.00      0.96        46
    
    avg / total       0.96      0.96      0.96       101
    
    0.960396039604


This Logistic Regression correctly identified all the non-winners in the test data, and only incorrectly identified winners in the test data 8% of the time.  I'd say it's a pretty good model!

### Visualize the results of the Logistic Regression PCA


```python
from matplotlib.colors import ListedColormap
%matplotlib inline
```

There are a lot of features in this data set, so let's use Principal Components Analysis to decompose the data into 2 dimensions so it's easy to visualize.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
```
Now let's apply Logistic Regression again on the decomposed data

```python
features_pca = pca.fit(features_norm).transform(features_norm)
pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(features_pca,y, test_size=0.2, random_state=0)
preds = LogisticRegression(C=5).fit(pca_X_train, pca_y_train).predict(pca_X_test)
```


```python
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
df1 = pd.DataFrame(pca_X_train)
df1['Result'] = pca_y_train
df1.plot(kind='scatter', x=0, y=1, c='Result', colormap = cmap_bold)
```


![Imgur](http://i.imgur.com/mogNyvc.png)


Above are the first and second principal components of the train data set, colored by the winners and non-winners.


```python
df2 = pd.DataFrame(pca_X_test)
df2['Predictions'] = preds
df2.plot(kind='scatter', x=0, y=1, c='Predictions', colormap = cmap_bold)
```

![Imgur](http://i.imgur.com/R1PBSCl.png)

Here's how the Logistic Regression splits the decomposed test data.  Let's compare it with the actual results of the test data.


```python
df2 = pd.DataFrame(pca_X_test)
df2['Actual'] = pca_y_test
df2.plot(kind='scatter', x=0, y=1, c='Actual', colormap = cmap_bold)
```

![Imgur](http://i.imgur.com/32VIWul.png)



The Logistic Regression did pretty well generalizing the data and sorting out the winners and non-winners of NaNoWriMo.  

Now let's try using a Decision Tree to classify winners and non-winners.


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
model_dt = DecisionTreeClassifier(max_depth=5)
print cross_val_score(model_dt, X_train, y_train, cv=10).mean()

model_dt.fit(X_train, y_train)
print confusion_matrix(y_test, model_dt.predict(X_test))
print classification_report(y_test,model_dt.predict(X_test))
print model_dt.score(X_test, y_test)
```

    0.969676360225
    [[50  5]
     [ 0 46]]
                 precision    recall  f1-score   support
    
              0       1.00      0.91      0.95        55
              1       0.90      1.00      0.95        46
    
    avg / total       0.96      0.95      0.95       101
    
    0.950495049505


The decision tree also performs pretty well in predicting winners and non-winners. Let's see what features it found to be most predictive.


```python
dt_importances = pd.DataFrame(zip(features.columns, model_dt.feature_importances_))
dt_importances.sort_values(1, ascending=False).head() # most to least predictive  
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
      <th>25</th>
      <td>SH Total</td>
      <td>0.795322</td>
    </tr>
    <tr>
      <th>23</th>
      <td>FH Total</td>
      <td>0.184662</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Expected Max Day</td>
      <td>0.015199</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Expected Avg Submission</td>
      <td>0.004818</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Member Length</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



SH Total, and FH Total are the most predictive features, but these are metrics collected after the current contest has started.  Let's build a model now with just information we have from past contests and see how that works.  
