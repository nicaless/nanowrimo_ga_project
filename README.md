# Predicting NaNoWriMo Winners

## Objective
Every year in November, writers all around the world participate in National Novel Writing Month (NaNoWriMo) and try to write 50,000 words of a novel within 30 days.  They track their word count progress on the NaNoWriMo website where they may also donate to the writing cause, join 'Regions' for writing camaraderie, and display the summary   of their novel in progress.  Those writers who write 50,000 words before the end of November are declared 'Winners'. 

My goal is to create a machine learning model that can predict whether a participating writer will be a NaNoWriMo 'winner' using data from the site.  

## Motivation

I love writing and I am enjoy participating in NaNoWriMo.  This idea stems from another personal project: creating my own [Word Count Tracker](http://nicaless.github.io/2015/11/09/My%20First%20Shiny%20App.html), similar to that on the NaNoWriMo website.  

INSERT PICS HERE

Visualizing writing progress can motivate one to write more and reach his or her writing goals!  I hope creating this predictive model may help other writers and future NaNoWriMo participants improve their writing strategies continue to write and finish their novels. 

### NaNoWriMo vocabulary
Some NaNoWriMo vocabulary to understand:

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

I used Kimono Labs to scrape most of the qualitative user data including usernames, whether they're a donor or even a volunteer [Municipal Liaison](http://nanowrimo.org/local-volunteers) for the site, if they're novels are [sponsored](http://nanowrimo.org/get-sponsored), and all the names of their past novels.  I was also able to get some quantitative data such as how long they've been a NaNoWriMo member, their lifetime word count, and what years they've participated.  

Below is a snapshot of Kimono Labs point-and-click interface to capture the data from a NaNoWriMo profile page.  
![Imgur](http://i.imgur.com/VcmuiS4.png)

However, I wasn't able to get the word count data from past NaNoWriMos using Kimono Labs.  That data is presented on each novel's stats page as a bar graph rendered by JavaScript.  Kimono can't parse JavaScript.  

![Imgur](http://i.imgur.com/ghONSFF.png)

I researched a few different ways to parse JavaScript using Python, but then I realized I only needed a single line of the JavaScript code that stored the data points for the graph.  I read the HTML document for each novel profile page as a regular text document and grabbed the line I needed.   

I also wanted to extract novel synopses and excerpts, but I ran into some difficulties using Kimono to grab the large amount of text from each novel profile page.  I decided it was time to switch tools.

![Imgur](http://i.imgur.com/0o62bzB.png)

With Beautiful Soup it was really easy to navigate the HTML structure of the novel profile page, and to find the tags and attributes for the text data I needed.

With all the data I needed, the next step was to process and aggregate all the information for analysis.


### Scraping Script Guide
The following are a description of the iPython scripts used to scrape data.

[GetCurrentContestStats](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/get_current_contest_stats.ipynb) - Utilizes NaNoWriMo API to get data from the most recent contest
[ScrapeNovelSynopses](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_novel_synopses.ipynb) - Uses Beautiful Soup to scrape each novel synopses 
[ScrapeNovelSynopses](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_novel_excerpt.ipynb) - Uses Beautiful Soup to scrape each novel excerpt
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

### Extracting Numeric Data from Novel Text Data

I had the following information on each of the novels of each writer.

[Novel Meta Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_meta_data.csv) - Contains the name of the novel, the writer, the genre, the final word count, daily average word count, and whether or not it was a winning novel
[Novel Word Count Info](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novels_wc_info.csv) - Basic statistics calculated for each novel

I merged these files on the novel name and also appended each novel's synopses and excerpt to create a [__novel_data.csv__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) file.

There is also a great deal of information in the text data for each novel - the genre, synopses, excerpt.  I hypothesize, if a writer is well-prepared for NaNoWriMo, they will have a clear genre chosen for their novel, and their novel profile will have a well-written synopses and excerpt - signs that their novel idea is fleshed out and they've done some planning before the contest starts.  

From the text data, I extracted numeric data such as number of words, unique words, paragraphs, and sentences in a synopses and excerpt.  I also calculated a reading score for the synopses and excerpt, and classified the genre of each novel as standard (fits into the usual novel genres such as Fiction, Historical, Young Adult) or non-standard (the novel hasn't been given a genre yet, it's a more obscure genre, or a combination of different genres). 

I then appended this data to the other novel data in another [__novel_features__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_features.csv) file.

### Aggregating Writer Data
In addition to their novels, I had the following raw data about each writer:

[Basic User Profile Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_basicinfo.csv) - A writer's username, their lifetime word count, how long the have been a NaNoWriMo member
[Fact Sheets](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_factsheet.csv) - Various information a writer could share about their age, occupation, location, hobbies, sponsorship, or role as a Municipal Liaison for NaNoWriMo
[Participation Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_participation.csv) - The past years a writer has participated in NaNoWriMo and whether they were winners or donors in that year.  

After a bit of cleaning, I merged the data in these files by writers' usernames.  

Now, I needed to somehow aggregate the major [__novel data__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) for each writer and merge it with the other writer data.

There were two different ways I aggregated the data.  In one way I took typical averages of the novel word count statistics.  In the other, I excluded novels created in the most current NaNoWriMo contest (November 2015).  I wanted to use these novels as the target of my predictions.  That is, I wanted to use the writers' past novels up to November 2014 to predict whether the novels of November 2015 would be 'winning novels' for the writer.  Thus, there are two similarly named 'user_summary' files.  

For the [__user_summary__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary.csv) file, certain statistics (eg. Expected Final Word Count, Expected Daily Average) take into account data from NaNoWriMo November 2015. 
The other file with [__'_no2015'__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary_no2015.csv) appended to the file name has the November 2015 information excluded from those statistics.  


### Processing Script Guide
The following are a description of the iPython scripts used to clean and process the raw data.

[FactSheetParser](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/fact_sheet_parser.ipynb) - Parses the raw Fact Sheets data 
[ParseMemberLength](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/parsememberlength.ipynb) - Cleans member length data in the raw Basic User Profile Data
[AppendParticipationData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata.ipynb)/[AppendParticipationData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata_negate2015.ipynb) - Two similar scripts that parse the raw Participation Data and appends results to other writer data (Basic Info, Fact Sheets)
[AggregateNovelStatsData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data.ipynb)/[AggregateNovelStatsData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data_negate2015.ipynb) - Two similar scripts that aggregate novel word count statistics and appends results to other writer data(Basic Info, Fact Sheets)
[AggregateFinalandDailyAvgs](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs.ipynb)/[AggregateFinalandDailyAvgs_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs%20no%202015.ipynb) - Two similar scripts that aggregate the final word count and daily average of novels and appends results to other writer data (Basic Info, Fact Sheets)
[CalculateTextFeaturesandReadingScore](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/calc_text_features_and_reading_score.ipynb) - Classified as novel's genre as standard or nonstandard, and extracted the number of words, unique words, sentences, paragraphs, and reading score of novel synopses.
[CalculateReadingScoreExcerpts](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/calc_flesch_reading_score-excerpts.ipynb) - Extracted the number of words, unique words, sentences, paragraphs, and reading score of novel excerpts.


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

__Excerpt__ The novel excerpt

### Novel Numeric Features - About the Data

Contains numeric data represeting each novel's genre, synopses, and excerpt.

There are 2122 rows and __ columns.

### Novel Numeric Features - Data Dictionary
ADD THE DATA DICTIONARY AND NOTE THAT SOME COLUMNS ARE DUPLICATES IN THE OTHER FILE SO YOU'RE NOT GOING TO RELIST THEM


## [Exploring the Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/exploratoryanalysis.ipynb)
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

### Text Features


```python
novel_features = pd.read_csv("../clean data/novel_features.csv", index_col = 0)
```


```python
print "The average number of words in a synopses is " + str(novel_features['num words'].mean())
print "There are " + str(novel_features['num words'].value_counts()[0]) + " novels without a synopsis"
winlose = novel_features[novel_features['num words'] >= 50]
winlose = winlose.groupby("Winner")
df = pd.DataFrame({'loss': winlose['num words'].get_group(0), 'win': winlose['num words'].get_group(1)})
df.plot(kind='hist', stacked=True, bins=[50,100,150,200,250,300,350,400,450,500,550,600], title = "Distribution of Words in Synopses")
```

    The average number of words in a synopses is 52.5930287329
    There are 729 novels without a synopsis

![Imgur](http://i.imgur.com/WVkKWlZ.png)


The average length in words of a synopses is 50 words, or about a few good sentences.  This is likely skewed by the fact that more than a third of the novels don't even have a synopsis.  There are few novels with synopses longer than 100 words, but as synopses get longer, it seems more likely that they belong to a winning novel.


```python
winlose = novel_features.groupby("Winner")
df = pd.DataFrame({'loss': winlose['fk score'].get_group(0), 'win': winlose['fk score'].get_group(1)})
df.plot(kind='hist', stacked=True, bins=[20,40,60,80,100,120], title = "Distribution of FK score")
```

![Imgur](http://i.imgur.com/sLHnooy.png)



```python
from scipy.stats import ttest_ind

ttest_ind(winlose['fk score'].get_group(0), winlose['fk score'].get_group(1))
```




    Ttest_indResult(statistic=-1.4376558464994371, pvalue=0.1506792394358735)



The Flesch-Kincaid reading scores look about normally distributed for this sample of novel synopses for both winners and non-winners, and since the pvalue from our t-test is greater than 10%, we cannot reject a null hypothesis that the winning and non-winning novels have equal averages of Flesch-Kincaid scores.  Flesch-Kincaid scores for a novel synopses are unlikely to be indicative of a winning novel.  


```python
df = novel_features[novel_features['fk score'] >= 0]
df = df[df['num words'] >= 50]
df.plot(kind='scatter', x='num sentences', y='fk score', c='Winner', colormap=cmap_bold, title = "FK Score vs Number of Sentences")
```


![Imgur](http://i.imgur.com/HV69yCg.png)


Trying to plot reading score of synopses against length of synopses produces this gobbled mess.  It may be hard trying to predict winning novels with these features...




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


## Applying Other Models (link to ipnb here)
including KMeans


## Modeling Novel Data (link to ipnb here)

## Genre Recommendation (link to ipnb here)

## Conclusion