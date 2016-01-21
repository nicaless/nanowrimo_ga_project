# Predicting NaNoWriMo Winners

## Objective
Every year in November, writers all around the world participate in National Novel Writing Month (NaNoWriMo) and try to write 50,000 words of a novel within 30 days.  They track their word count progress on the NaNoWriMo website where they may also donate to the writing cause, join 'Regions' for writing camaraderie, and display the summary of their novel in progress.  Those writers who write 50,000 words before the end of November are declared 'Winners'. 

My goal is to create a machine learning model that can predict whether a participating writer will be a NaNoWriMo 'winner' using data from the site.  

## Motivation

I love writing and I am enjoy participating in NaNoWriMo.  This idea stems from another personal project to create my own [Word Count Tracker](http://nicaless.github.io/2015/11/09/My%20First%20Shiny%20App.html).  It started off as simply an exploratory endeavor and a way to practive visualizations in R and become familiar with R's Shiny package for interactive visualization applications.  It will be incredibly fun to recreate and extend this project using Python.  I hope to find new insights in the data by creating this predictive model.  I also hope such a model may be able to help other writers and future participants in NaNoWriMo improve their writing strategies and become motivated to continue to write. 

### NaNoWriMo vocabulary
Some NaNoWriMo vocabulary (as defined by me!) to understand:

__Writer__ - A NaNoWriMo.org user that is participating in a current NaNoWriMo contest.
__Win__ - When a writer reaches 50,000 word count goal for their novel and validates this word count with the NaNoWriMo website. 
__Word Count/Word Count Submission__ - For a novel or a submission to that novel, the number of words recorded to have been written 
__Submission__ - The act of updating the word count for a novel. During a contest, if there is no update for a novel on a given day, the word count submission for that novel is recorded as 0 and the total word count for a novel remains the same. NOTE: A writer my update the word count for their novel multiple times a day. The site will not record the updates until the end of the data.  The aggregate of these updates is the submission.      
__Contest__ -  A NaNoWriMo event.  That is, when the NaNoWriMo site opens and writers may create a novel profile and begin writing and adding submissions.  
__Donation/Donor__ - If a user makes a monetary donation to the NaNoWriMo organization and their mission, they are marked as a 'donor' on the site.  NaNoWriMo does not disclose the amount the user donated, just that they are a donor.  NOTE: A user may donate without being a writer. But for the purposes of this project, those users don't exist in this data set :)     
__Municipal Liaison__ - Taken from the NaNoWriMo website: "Municipal Liaisons (MLs) are volunteers who add a vibrant, real-world aspect to NaNoWriMo festivities all over the world." These writers are particularly involved NaNoWriMo users :D 
__Sponsorship__ - Writers may have their novels sponsored, with the sponsor money going to further the NaNoWriMo mission.  
__Novel__ - A writer's 'entry' in the NaNoWriMo contest - the thing they commit to writing during the contest.  NOTE: 'Novels' may not actually be novels.  Writers may choose to write memoirs, non-fiction, movie scripts, etc.   


## The Data

The data I will use to construct this model is user data and novel data from the website.  This includes usernames, novel titles, word count, and 'Winner' labels.

## Scraping NaNoWriMo Data

I created a script utilizing the site's Word Count API to get word count submission history. 

The trouble is, the NaNoWriMo API, as far as I know, only gets data from the most recent contest, in this case, November 2015.  This was not enough to make much of an interesting model.  

Other data I wanted to incorporate in the model include a user's past daily word count averages, number of novels started, novel synopses, and whether or not they've donated to the NaNoWriMo cause.  

Luckily, all the data I wanted was available on the NaNoWriMo website, but I wasn't about to click through 500+ user profiles manually entering information into a spreadsheet to get all of it.  

I used Kimono to scrape most of the qualitative user data including usernames, whether they're a donor or even a volunteer [Municipal Liaison](http://nanowrimo.org/local-volunteers) for the site, if they're novels are [sponsored](http://nanowrimo.org/get-sponsored), and all the names for their past novels.  I was also able to get some quantitative data like how long they've been a NaNoWriMo member, their lifetime word count, and what years they've participated.  

However, I wasn't able to get the word count data from past NaNoWriMos using Kimono Labs.  That data is presented on each novel's stats page as a bar graph rendered by JavaScript.  Kimono can't parse JavaScript.  

I researched a few different ways to parse JavaScript using Python, but then I realized I only needed a single line of the JavaScript code.  So I just read the HTML document for each novel stats page as a regular text document and grabbed the line I needed.   

I also wanted to extract novel synopses, but I ran into some difficulties using Kimono to grab the large amount of text from each novel stats page.  I decided it was time to switch tools.

With Beautiful Soup it was really easy to navigate the HTML structure of the novel stats page, and to find the tags and attributes for the text data I needed.

With all the data I needed, the next step was to process and aggregate all the information for analysis.


### Scraping Script Guide
The following are a description of the iPython scripts used to scrape data.

[GetCurrentContestStats](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/get_current_contest_stats.ipynb) - Utilizes NaNoWriMo API to get data from the most recent contest
[ScrapeNovelSynopses](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_novel_synopses.ipynb) - Uses Beautiful Soup to scrape each novel synopses 
[ScrapeWCSubmissions](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_wc_submissions.ipynb) - Parses HTML file for a JavaScript variable that contains information about daily word count submission for each novel 

### Raw Data Guide 

 Data | Description | Source
---|---|---
[User Names](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_names.csv) | A list of writer's usernames | Hand collected
[Novel Pages](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_pages.csv) | A list of novels by the selected writers | Kimono Labs API
[Novel WC Info](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novels_wc_info.csv) | Word count stats for each of the novels | The [ScrapeWCSubmissions](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_wc_submissions.ipynb) script
[Novel Names, Urls, Dates](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_names_urls_dates.csv) | The novels with their respective NaNoWriMo page urls and the date they were entered into a NaNoWriMo contest | Kimono Labs API
[Novel Meta Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_meta_data.csv) | Contains more information about the novels | Kimono Labs API  
[Basic User Profile Information](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_basicinfo.csv) | A writer's username, their lifetime word count, how long the have been a NaNoWriMo member | Kimono Labs API
[Fact Sheets](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_factsheet.csv) | Various information a writer could share about their age, occupation, location, hobbies, sponsorship, or role as a Municipal Liaison for NaNoWriMo |  Kimono Labs API
[Participation Information](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_participation.csv) | The past years a writer has participated in NaNoWriMo and whether they were winners or donors in that year | Kimono Labs API
 
## The Data Processing Process

After scraping all the data, the task at hand was to aggregate the information.   

In three separate files, I had the following raw data about each writer:

[Basic User Profile Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_basicinfo.csv) - A writer's username, their lifetime word count, how long the have been a NaNoWriMo member
[Fact Sheets](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_factsheet.csv) - Various information a writer could share about their age, occupation, location, hobbies, sponsorship, or role as a Municipal Liaison for NaNoWriMo
[Participation Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_participation.csv) - The past years a writer has participated in NaNoWriMo and whether they were winners or donors in that year.  

After a bit of cleaning, I merged the data in these files by writers' usernames.  

In addition to information about each writer, I also had information on each of the novels of each writer.

[Novel Meta Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_meta_data.csv) - Contains the name of the novel, the writer, the genre, the final word count, daily average word count, and whether or not it was a winning novel
[Novel Word Count Info](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novels_wc_info.csv) - Basic statistics calculated for each novel

I merged these files on the novel name and also appended each novel's synopses to create a final [__novel_data.csv__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) file.

Now, I needed to somehow aggregate the novel data for each writer and merge it with the other writer data.

There were two different ways I aggregated the data.  In one way I took typical averages of the novel word count statistics.  In the other, I excluded novels created in the most current NaNoWriMo contest (November 2015).  I wanted to use these novels as the target of my predictions.  That is, I wanted to use the writers' past novels up to November 2014 to predict whether the novels of November 2015 would be 'winning novels' for the writer.  Thus, there are two similarly named 'user_summary' files.  

For the [__user_summary__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary.csv) file, certain statistics (eg. Expected Final Word Count, Expected Daily Average) take into account data from NaNoWriMo November 2015. 
The other file with [__'_no2015'__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary_no2015.csv) appended to the file name has this information excluded from those statistics.  

For the purposes of this project, I will use the latter file for analysis and model training.

### Processing Script Guide
The following are a description of the iPython scripts used to clean and process the raw data.

[FactSheetParser](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/fact_sheet_parser.ipynb) - Parses the raw Fact Sheets data 
[ParseMemberLength](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/parsememberlength.ipynb) - Cleans member length data in the raw Basic User Profile Data
[AppendParticipationData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata.ipynb)/[AppendParticipationData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata_negate2015.ipynb) - Two similar scripts that parse the raw Participation Data and appends results to other writer data (Basic Info, Fact Sheets)
[AggregateNovelStatsData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data.ipynb)/[AggregateNovelStatsData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data_negate2015.ipynb) - Two similar scripts that aggregate novel word count statistics and appends results to other writer data(Basic Info, Fact Sheets)
[AggregateFinalandDailyAvgs](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs.ipynb)/[AggregateFinalandDailyAvgs_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs%20no%202015.ipynb) - Two similar scripts that aggregate the final word count and daily average of novels and appends results to other writer data (Basic Info, Fact Sheets)


## Data Dictionaries

### Writers - About the Data

Contains basic profile information about each writer and their past NaNoWriMo statistics.

There are 501 rows and 41 columns.

The data may be found [here](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary_no2015.csv).

### Writers - Data Dictionary
__Writer Name__ - The writer's NaNoWriMo username

__Member Length__ - The number of years a writer has been a NaNoWriMo user

__LifetimeWordCount__ - The total number of words a writer has written over all NaNoWriMo contests

__url__ - The url to the writer's profile on NaNoWriMo.org

__Age__ - The age of the writer

__Birthday__ - The birthday of the writer

__Favorite books or authors__ - The writer's recorded favorite books or authors

__Favorite noveling music__ - The writer's favorite music to listen to while writing

__Hobbies__ - The writer's recorded hobbies

__Location__ - The location from where the writer is writing

__Occupation__ - The writer's recorded occupation

__Primary Role__ - If the writer is a "Municipal Liaison" for NaNoWriMo, it is recorded here

__Sponsorship URL__ - If the writer's novel is sponsored, a sponsorship url is recorded here

__Expected Final Word Count__ - The average of the final word count for all a writer's novels

__Expected Daily Average__ - The average of the daily average word count for all a writer's novels, calculated from using 

__CURRENT WINNER__ - Indicates whether the writer is a winner of the "current" or "next" NaNoWriMo (November 2015)

__Current Donor__  - Indicates whether the writer is a donor of the "current" or "next" NaNoWriMo (November 2015)

__Wins__ - The number of past wins for a writer.  Wins cannot be greater than Participated.

__Donations__ - The number of past donations for a writer.  Donations cannot be greater than Participated.

__Participated__ - The number of past NaNoWriMo contests in which the writer was a participant

__Consecutive Donor__ - The maximum number of consecutive contests for which the writer has donated

__Consecutive Wins__ - The maximum number of consecutive contests for which the writer has won 

__Consecutive Part__ - The maximum number of consecutive contests for which the writer has participated

__Part Years__ - A list of years for which the writer has participated in NaNoWriMo

__Win Years__ - A list of years for which the writer has won 

__Donor Years__ - A list of years for which the writer has donated

__Num Novels__ - The number of novels which a writer has entered into NaNoWriMo

__Expected Num Submissions__ - The average. over all a writer's novels, of the number of word count 
submissions entered for a novel

__Expected Avg Submission__ - The average. over all a writer's novels, of the average number of words entered in all word count submissions for a novel

__Expected Min Submission__ - The average, over all a writer's novels, of the minimum number of words entered in all word count submissions for a novel

__Expected Min Day__ - The average day (from 1-30), over all contests a writer participated, on which the writer entered the minimum number of words

__Expected Max Submission__ - The average, over all a writer's novels, of the maximum number of words entered in all word count submissions for a novel

__Expected Max Day__ - The average day (from 1-30), over all contests a writer participated, on which the writer entered the maximum number of words

__Expected Std Submissions__ - The average, over all a writer's novels, of the standard deviation of the number of words entered for all word count submissions for a novel

__Expected Consec Subs__ - The average, over all a writer's novels, of the number of consecutive submissions (at least 2 submissions in a row) entered for a novel

__FW Total__ - For the current NaNoWriMo, the total word count of a novel in the first week of the contest

__FW Sub__ - For the current NaNoWriMo, the number of word count submissions to a novel in the first week of the contest

__FH Total__ - For the current NaNoWriMo, the total word count of a novel in the first half of the contest

__FH Sub__ - For the current NaNoWriMo, the number of word count submissions to a novel in the first half of the contest

__SH Total__ - For the current NaNoWriMo, the total word count of a novel in the second half of the contest

__SH Sub__ - For the current NaNoWriMo, the number of word count submissions to a novel in the second half of the contest    

### Novels - About the Data

Contains basic profile information about each novel and their word count statistics.

There are 2122 rows and 9 columns.

The data may be found [here](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv).

### Novels - Data Dictionary

__Writer Name__ - The writer of the novel

__Novel Name__ - The title of the novel

__Genre__ - The genre of the novel

__Final Word Count__ The final recorded word count for the novel

__Daily Average__ The average recorded word count of the novel over the 30 day period of its contest

__Winner__  Indicates whether the novel is a winning novel (reached 50,000 words) during its contest 

__Synopses__ The novel synopses

__url__ The url of the novel's stats page

__Novel Date__ The date of the contest for which the novel was written


## Exploring the Data
After I had constructed the data, I proceeded with some preliminary explorations with Python and matplotlib.


```python
import pandas as pd
import numpy as np
import warnings
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')
%matplotlib inline
```

### Let's take a look at the Writer data


```python
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



### Wins for Losses for NaNoWriMo 2015


```python
# ratio of wins to losses
winners = len(writers[writers['CURRENT WINNER'] == 1])
nonwinners = len(writers[writers['CURRENT WINNER'] == 0])
print "There are " + str(winners) + " winners out of " + str(len(writers[writers['CURRENT WINNER']])) + " writers"
print "There are " + str(nonwinners) + " nonwinners out of " + str(len(writers[writers['CURRENT WINNER']])) + " writers"
print "Therefore there is a " + str( ( float(winners) / len(writers[writers['CURRENT WINNER']]) )) + "% of winning"
print "The ratio of winners to nonwinners is " + str( float(winners) / nonwinners) 
```

    There are 219 winners out of 501 writers
    There are 282 nonwinners out of 501 writers
    Therefore there is a 0.437125748503% of winning
    The ratio of winners to nonwinners is 0.776595744681


At first glance, winning is almost a coin-toss.  One can have an almost 50/50 chance at guessing whether a writer will win NaNoWriMo 2015.

### Interesting Correlations

__Lifetime Word Count vs Member Length__


```python
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
writers.plot(kind='scatter', x='Member Length', y='LifetimeWordCount', c='CURRENT WINNER', colormap = cmap_bold, title = "LifetimeWordCount vs Member Length")
```


![Imgur](http://i.imgur.com/OM5Wpyc.png)


Few writers have a written more than 1,000,000 words (or 20 NaNoWriMo winning novels) over the course of their NaNoWriMo lifetime.  The density of nonwinners for NaNoWriMo 2015 decreases as Member Length increases, and a higher Lifetime Word Count indicates higher likelihood of winning.  It makes sense that the longer one writes (Member Length) and the more words one writes (Lifetime word Count) makes one more likely to reach the NaNoWriMo writing goal.

__Expected Avg Submission vs Expected Daily Average__


```python
df = writers[writers['Expected Avg Submission'] <= 15000]
df.plot(kind='scatter', x='Expected Daily Average', y='Expected Avg Submission', c='CURRENT WINNER', colormap=cmap_bold, title = "Expected Avg Submission vs Expected Daily Average")
```


![Imgur](http://i.imgur.com/VbtG1FX.png)


It almost looks like there are clusters.  If Expected Daily Average => Expected Avg Submission, a writer is more likely to win.  It's worth noting that the minimum daily average needed to win a NaNoWriMo contest is about 1,666 (50,000 words / 30 days).

__Number of Wins vs Number of times participated__


```python
df = writers[writers['Wins'] <= 30]
df.plot(kind='scatter', x='Participated', y='Wins', c='CURRENT WINNER', colormap=cmap_bold, title = "Num Wins vs Num Participated")

```


![Imgur](http://i.imgur.com/lBqjeFm.png)


It looks like there may be possible clusters here as well.  Writers who have already had more than 5 wins are very likely to win again.  Also writers who have participated more than 5-10 times have better chances of winning as well.

__Expected Daily Average vs Expected Num Submissions__


```python
df = writers[writers['Expected Daily Average'] <= 10000]
df.plot(kind='scatter', x='Expected Num Submissions', y='Expected Daily Average', c='CURRENT WINNER', colormap=cmap_bold, title = "Expected Daily Average vs Expected Num Submissions")

```


![Imgur](http://i.imgur.com/BHBCI0G.png)


Many writers seem to cluster around an Expected Daily Average of 1500-2000.  Remember, 1,666 is the minimum daily average to win a NaNoWriMo contest.  Of course, the higher an Expected Daily Average, the more likely a writer is to win the upcoming contest.  Also interesting is how the density of nonwinners decreases as Expected Num Submissions increases, so higher Expected Num Submissions may also be indicative of winning.   

__Distribution Word Count Submissions in early weeks of a contest__

I wanted to look retrospectively at the latest NaNoWriMo contest and see how winners can be predicted as early as the first week or two weeks of a contest


```python
winlose = writers.groupby("CURRENT WINNER")
df = pd.DataFrame({'loss': winlose['FW Sub'].get_group(0), 'win': winlose['FW Sub'].get_group(1)})
df.plot(kind='hist', stacked=True, title = "distribution of number ofsubmissions in first week of contest")
```

![Imgur](http://i.imgur.com/TQkf7Mb.png)

```python
winlose = writers.groupby("CURRENT WINNER")
df = pd.DataFrame({'loss': winlose['FH Sub'].get_group(0), 'win': winlose['FH Sub'].get_group(1)})
df.plot(kind='hist', stacked=True, title = "distribution of number of submissions in first half of contest")
```

![Imgur](http://i.imgur.com/LRBKpWn.png)

As expected, writers who submit more often in the early weeks are more likely to win.

__Average First Week Submissions vs Expected Daily Average__

```python
df = writers
df['FW Avg'] = df['FW Total'] / 7
df.plot(kind='scatter', x='FW Avg', y='Expected Daily Average', c='CURRENT WINNER', colormap=cmap_bold, title = "Average First Week Submission vs Expected Num Submissions")

```

![Imgur](http://i.imgur.com/mWKMKoy.png)


Additionally, writers whose daily average in the first week => than the Expected Daily Average of their past novels are more likely to win.

__Does Municipal Liaison or having a novel sponsored have effect on winning?__


```python
# Convert to binary
writers['Primary Role'][writers['Primary Role'] == 'Municipal Liaison'] = 1
writers['Primary Role'][writers['Primary Role'] != 1] = 0
writers['Sponsorship URL'].fillna(0, inplace=True)
writers['Sponsorship URL'][writers['Sponsorship URL'] != 0] = 1
```


```python
winlose = writers.groupby("CURRENT WINNER")
df = pd.DataFrame({'loss': winlose['Primary Role'].get_group(0), 'win': winlose['Primary Role'].get_group(1)})
df.plot(kind='hist', stacked=True, title = "Distribution of winners and nonwinners for Municipal Liaisons")
```



![Imgur](http://i.imgur.com/b4jSjo2.png)



```python
winlose = writers.groupby("CURRENT WINNER")
df = pd.DataFrame({'loss': winlose['Sponsorship URL'].get_group(0), 'win': winlose['Sponsorship URL'].get_group(1)})
df.plot(kind='hist', stacked=True, title = "Distribution of winners and nonwinners for those with sponsorships")
```


![Imgur](http://i.imgur.com/zMGGwTa.png)


It definitely seems like one is more likely to win if they are a Municipal Liaison of if their novel is sponsored!


```python
sponsors = writers[writers['Sponsorship URL'] == 1]
winners = len(sponsors[sponsors['CURRENT WINNER'] == 1])
nonwinners = len(sponsors[sponsors['CURRENT WINNER'] == 0])
print "The ratio of winners to nonwinners for those with Sponsors is " + str( float(winners) / nonwinners)
```

The ratio of winners to nonwinners for those with Sponsors is 2.0



```python
mls = writers[writers['Primary Role'] == 1]
winners = len(mls[mls['CURRENT WINNER'] == 1])
nonwinners = len(mls[mls['CURRENT WINNER'] == 0])
print "The ratio of winners to nonwinners for those who are MLs is " + str( float(winners) / nonwinners)
```

The ratio of winners to nonwinners for those who are MLs is 5.85714285714


### Now let's look at the novel data


```python
novels = pd.read_csv("../clean data/novel_data.csv", index_col=0)
novels.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Writer Name</th>
      <th>Novel Name</th>
      <th>Genre</th>
      <th>Final Word Count</th>
      <th>Daily Average</th>
      <th>Winner</th>
      <th>Synopses</th>
      <th>url</th>
      <th>Novel Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nicaless</td>
      <td>Novel: Lauren's Birthday</td>
      <td>Genre: Young Adult</td>
      <td>24229</td>
      <td>807</td>
      <td>0</td>
      <td>\n&lt;p&gt;&lt;/p&gt;\n</td>
      <td>http://nanowrimo.org/participants/nicaless/nov...</td>
      <td>November 2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nicaless</td>
      <td>Novel: A Mystery in the Kingdom of Aermon</td>
      <td>Genre: Fantasy</td>
      <td>50919</td>
      <td>1,697</td>
      <td>1</td>
      <td>\n&lt;p&gt;Hitoshi is appointed the youngest Judge a...</td>
      <td>http://nanowrimo.org/participants/nicaless/nov...</td>
      <td>November 2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rachel B. Moore</td>
      <td>Novel: Finding Fortunato</td>
      <td>Genre: Literary</td>
      <td>50603</td>
      <td>1,686</td>
      <td>1</td>
      <td>\n&lt;p&gt;Sam and Anna Gold and their newly adoptiv...</td>
      <td>http://nanowrimo.org/participants/rachel-b-moo...</td>
      <td>November 2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rachel B. Moore</td>
      <td>Novel: The Residency</td>
      <td>Genre: Literary</td>
      <td>50425</td>
      <td>1,680</td>
      <td>1</td>
      <td>\n&lt;p&gt;It's every writer's dream - an all-expens...</td>
      <td>http://nanowrimo.org/participants/rachel-b-moo...</td>
      <td>November 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rachel B. Moore</td>
      <td>Novel: The Jew From Fortunato</td>
      <td>Genre: Literary Fiction</td>
      <td>41447</td>
      <td>1,381</td>
      <td>0</td>
      <td>\n&lt;p&gt;20-something Andre Levinsky is a fish out...</td>
      <td>http://nanowrimo.org/participants/rachel-b-moo...</td>
      <td>November 2013</td>
    </tr>
  </tbody>
</table>
</div>

__Overall Wins and Losses__


```python
winners = len(novels[novels['Winner'] == 1])
nonwinners = len(novels[novels['Winner'] == 0])
print "The total number of novels written is " + str( winners + nonwinners)
print str(winners) + " are winners"
print str(nonwinners) + " are not winners"
print "Therefore " + str( (float(winners) / (nonwinners + winners) ) * 100 ) + "% are winners"
print "The ratio of winners to nonwinners is " + str( float(winners) / nonwinners)

```

The total number of novels written is 2123
1333 are winners
790 are not winners
Therefore 62.78850683% are winners
The ratio of winners to nonwinners is 1.68734177215


It's interesting that there are more winning novels than nonwinning novels while there are more winning writers for the most recent NaNoWriMo than there are nonwinning writers.  But this makes sense because writers who write more novels are more likely to have their novels reach the 50,000 word goal.


## Predicting NaNoWriMo winners with Logistic Regression

As the variable I want to predict is binary (1 if a writer is a winner, 0 if otherwise) I decided to use a logistic regression as my prediction model.  

From my preliminary explorations of the numerical data for writers, it seemed like some important features to include were:

- Member Length
- LifetimeWordCount
- Primary Role (whether or not a writer is a Municipal Liaison) 
- Sponsorship URL (whether or not a writer's novel is sponsored)
- Expected Daily Average
- Expected AvgSubmission
- Expected Number of Submissions
- Number of Wins
- Number of times they have donated
- Number of times they have participated  

I decided these would be a good start to building the model.


```python
# import the data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

writers = pd.read_csv("../clean data/user_summary_no2015.csv", index_col=0)
```


```python
# convert primary role and sponsorship url to binary vars
writers['Primary Role'][writers['Primary Role'] == 'Municipal Liaison'] = 1
writers['Primary Role'][writers['Primary Role'] != 1] = 0

writers['Sponsorship URL'].fillna(0, inplace=True)
writers['Sponsorship URL'][writers['Sponsorship URL'] != 0] = 1
```


```python
# let's keep the columns that matter
features = writers[['Member Length','LifetimeWordCount','Primary Role','Sponsorship URL','Expected Daily Average','Expected Avg Submission','Expected Num Submissions','Wins','Donations','Participated']]
```


```python
features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member Length</th>
      <th>LifetimeWordCount</th>
      <th>Primary Role</th>
      <th>Sponsorship URL</th>
      <th>Expected Daily Average</th>
      <th>Expected Avg Submission</th>
      <th>Expected Num Submissions</th>
      <th>Wins</th>
      <th>Donations</th>
      <th>Participated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>50919</td>
      <td>0</td>
      <td>0</td>
      <td>1697.300000</td>
      <td>3637.071429</td>
      <td>14.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>478090</td>
      <td>1</td>
      <td>0</td>
      <td>1593.633333</td>
      <td>918.057453</td>
      <td>8.300000</td>
      <td>8</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>475500</td>
      <td>1</td>
      <td>1</td>
      <td>1440.909091</td>
      <td>822.780595</td>
      <td>9.272727</td>
      <td>7</td>
      <td>7</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>30428</td>
      <td>0</td>
      <td>0</td>
      <td>507.133333</td>
      <td>678.318083</td>
      <td>22.000000</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = writers['CURRENT WINNER'].values
```


```python
# inputting 0 for users without prior data for daily avg, avg submission, num submissions
features['Expected Daily Average'].fillna(0, inplace=True)
features['Expected Avg Submission'].fillna(0, inplace=True)
features['Expected Num Submissions'].fillna(0, inplace=True)
features.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member Length</th>
      <th>LifetimeWordCount</th>
      <th>Expected Daily Average</th>
      <th>Expected Avg Submission</th>
      <th>Expected Num Submissions</th>
      <th>Wins</th>
      <th>Donations</th>
      <th>Participated</th>
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
    </tr>
    <tr>
      <th>mean</th>
      <td>4.212575</td>
      <td>172552.676647</td>
      <td>1214.277073</td>
      <td>1708.026777</td>
      <td>10.826177</td>
      <td>2.606786</td>
      <td>1.421158</td>
      <td>3.656687</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.255209</td>
      <td>329113.331830</td>
      <td>1459.407277</td>
      <td>2053.622361</td>
      <td>8.520344</td>
      <td>4.651782</td>
      <td>3.044384</td>
      <td>4.899582</td>
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
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>9818.000000</td>
      <td>248.108333</td>
      <td>362.750000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>93385.000000</td>
      <td>1253.144444</td>
      <td>1446.652778</td>
      <td>10.250000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>206482.000000</td>
      <td>1691.140000</td>
      <td>2213.520000</td>
      <td>17.333333</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.000000</td>
      <td>4562712.000000</td>
      <td>21727.200000</td>
      <td>20869.236584</td>
      <td>30.000000</td>
      <td>52.000000</td>
      <td>36.000000</td>
      <td>52.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

```

### Normalize data


```python
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)
features_norm[1]
```




    array([ 1.77967343,  0.92929298,  3.07205143, -0.19304684,  0.26019839,
           -0.38505565, -0.29678396,  1.16054543,  2.1631367 ,  1.29595831])



### Apply Logistic Regression


```python
model_lr = LogisticRegression(C=5)
cross_val_score(model_lr,features_norm,y,cv=3).mean()
```




    0.69660678642714569



Remember, by just 'guessing' we can predict that a writer will win NaNoWriMo with 44% chance, or about 50/50.  That means, this logistic regression model does almost 25% better!


```python
model_lr = LogisticRegression(C=5).fit(features_norm, y)
coefficients=model_lr.coef_.ravel()
coefficients
```




    array([-0.05032326,  1.89666991,  0.57005087,  0.01114819, -0.01187803,
            0.11182389,  0.07800505,  1.83989233, -0.22167569, -1.89802147])

## Next Steps

Improving the model.  Feature engineering.  Particularly I want to include the text data available for the novels (genre, synopses).

Find

- whether or not there is a synopses
- number of words in synopses
- number of unique words
- number of number of paragraphs
- Flesh-Kincaid reading score for synopses

Then see if these features are indicative of a novel 'winning'.  Then I can use this information to predict whether a writer will win. 