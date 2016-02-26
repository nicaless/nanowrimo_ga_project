# Predicting NaNoWriMo Winners

- [Table of Contents](#)
	- [Objective](#)
	- [Motivation](#)
	- [The Data](#)
		- [NaNoWriMo vocabulary](#)
		- [Scraping NaNoWriMo Data](#)
			- [Scraping Script Guide](#)
			- [Raw Data Guide](#)
		- [The Data Processing Process](#)
			- [Extracting Numeric Data from Novel Text Data](#)
			- [Aggregating Writer Data](#)
			- [Processing Script Guide](#)
		- [Data Dictionaries](#)
			- [Writers - About the Data](#)
			- [Writers - Data Dictionary](#)
			- [Novels - About the Data](#)
			- [Novels - Data Dictionary](#)
			- [Novel Numeric Features - About the Data](#)
			- [Novel Numeric Features - Data Dictionary](#)
	- [Exploring the Data](#)
		- [Let's take a look at the Writer data](#)
		- [Now let's look at the novel data](#)
		- [Text Features](#)
	- [Logistic Regression](#)
		- [Normalize data](#)
		- [Apply Logistic Regression](#)
		- [Visualize the results of the Logistic Regression with PCA](#)
	- [Using Fewer Feaures and Applying Other Models](#)
		- [Re-apply Logistic Regression](#)
		- [Naive Bayes](#)
		- [SVM](#)
		- [Decision Tree](#)
		- [Random Forests](#)
	- [Modeling Novel Data](#)
		- [Logistic Regression](#)
		- [K Neighbors](#)
		- [Naive Bayes](#)
		- [Decision Tree](#)
		- [Random Forest](#)
		- [Support Vector Machine](#)
	- [Clusters of Writers](#)
		- [K Means](#)
		- [Silhouette Scores](#)
	- [Genre Recommendation](#)
	- [Conclusion](#)
		- [Next Steps](#)


## Objective
Every year in November, writers all around the world participate in National Novel Writing Month (NaNoWriMo) and try to write 50,000 words of a novel within 30 days.  They track their word count progress on the NaNoWriMo website where they may also donate to the writing cause, join 'Regions' for writing camaraderie, and display the summary   of their novel in progress.  Those writers who write 50,000 words before the end of November are declared 'Winners'. 

My goal is to create a machine learning model that can predict whether a participating writer will be a NaNoWriMo 'winner' using data from the site.  

## Motivation

I love writing and I am enjoy participating in NaNoWriMo.  This idea stems from another personal project: creating my own [Word Count Tracker](http://nicaless.github.io/2015/11/09/My%20First%20Shiny%20App.html) that would track how much I write over time, similar to that of the cumulative word count graph displayed on each writer's novel profile every NaNoWrimo.  

PICTURE?

I wanted to take it a step further and also visualize the aggregate word count progress of a region and the whole site.

![Imgur](http://i.imgur.com/XXub2OU.png)
![Imgur](http://i.imgur.com/yp0l5VC.png)
PICTURE


Visualizing writing progress can motivate one to write more and reach his or her writing goals!  I hope creating this predictive model may help other writers and future NaNoWriMo participants improve their writing strategies continue to write and finish their novels. 


## The Data

The data I will use to construct this model is user data and novel data from the website.  This includes usernames, novel titles, word count, and 'Winner' labels.

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


### Scraping NaNoWriMo Data

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


#### Scraping Script Guide
The following are a description of the iPython scripts used to scrape data.

[GetCurrentContestStats](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/get_current_contest_stats.ipynb) - Utilizes NaNoWriMo API to get data from the most recent contest
[ScrapeNovelSynopses](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_novel_synopses.ipynb) - Uses Beautiful Soup to scrape each novel synopses 
[ScrapeNovelSynopses](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_novel_excerpt.ipynb) - Uses Beautiful Soup to scrape each novel excerpt
[ScrapeWCSubmissions](https://github.com/nicaless/nanowrimo_ga_project/blob/master/scrape/scrape_wc_submissions.ipynb) - Parses HTML file for a JavaScript variable that contains information about daily word count submission for each novel 

#### Raw Data Guide 

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
 
### The Data Processing Process

After scraping all the data, the task at hand was to aggregate the information.   

#### Extracting Numeric Data from Novel Text Data

I had the following information on each of the novels of each writer.

[Novel Meta Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novel_meta_data.csv) - Contains the name of the novel, the writer, the genre, the final word count, daily average word count, and whether or not it was a winning novel
[Novel Word Count Info](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/novels_wc_info.csv) - Basic statistics calculated for each novel

I merged these files on the novel name and also appended each novel's synopses and excerpt to create a [__novel_data.csv__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) file.

There is also a great deal of information in the text data for each novel - the genre, synopses, excerpt.  I hypothesize, if a writer is well-prepared for NaNoWriMo, they will have a clear genre chosen for their novel, and their novel profile will have a well-written synopses and excerpt - signs that their novel idea is fleshed out and they've done some planning before the contest starts.  

From the text data, I extracted numeric data such as number of words, unique words, paragraphs, and sentences in a synopses and excerpt.  I also calculated a reading score for the synopses and excerpt, and classified the genre of each novel as standard (fits into the usual novel genres such as Fiction, Historical, Young Adult) or non-standard (the novel hasn't been given a genre yet, it's a more obscure genre, or a combination of different genres). 

I then appended this data to the other novel data in another [__novel_features__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_features.csv) file.

#### Aggregating Writer Data
In addition to their novels, I had the following raw data about each writer:

[Basic User Profile Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_basicinfo.csv) - A writer's username, their lifetime word count, how long the have been a NaNoWriMo member
[Fact Sheets](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_factsheet.csv) - Various information a writer could share about their age, occupation, location, hobbies, sponsorship, or role as a Municipal Liaison for NaNoWriMo
[Participation Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/rawdata/user_profiles_participation.csv) - The past years a writer has participated in NaNoWriMo and whether they were winners or donors in that year.  

After a bit of cleaning, I merged the data in these files by writers' usernames.  

Now, I needed to somehow aggregate the major [__novel data__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) for each writer and merge it with the other writer data.

There were two different ways I aggregated the data.  In one way I took typical averages of the novel word count statistics.  In the other, I excluded novels created in the most current NaNoWriMo contest (November 2015).  I wanted to use these novels as the target of my predictions.  That is, I wanted to use the writers' past novels up to November 2014 to predict whether the novels of November 2015 would be 'winning novels' for the writer.  Thus, there are two similarly named 'user_summary' files.  

For the [__user_summary__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary.csv) file, certain statistics (eg. Expected Final Word Count, Expected Daily Average) take into account data from NaNoWriMo November 2015. 
The other file with [__'_no2015'__](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary_no2015.csv) appended to the file name has the November 2015 information excluded from those statistics.  


#### Processing Script Guide
The following are a description of the iPython scripts used to clean and process the raw data.

[FactSheetParser](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/fact_sheet_parser.ipynb) - Parses the raw Fact Sheets data 
[ParseMemberLength](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/parsememberlength.ipynb) - Cleans member length data in the raw Basic User Profile Data
[AppendParticipationData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata.ipynb)/[AppendParticipationData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/appendparticipationdata_negate2015.ipynb) - Two similar scripts that parse the raw Participation Data and appends results to other writer data (Basic Info, Fact Sheets)
[AggregateNovelStatsData](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data.ipynb)/[AggregateNovelStatsData_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate_novel_stats_data_negate2015.ipynb) - Two similar scripts that aggregate novel word count statistics and appends results to other writer data(Basic Info, Fact Sheets)
[AggregateFinalandDailyAvgs](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs.ipynb)/[AggregateFinalandDailyAvgs_negate2015](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/aggregate%20final%20and%20daily%20avgs%20no%202015.ipynb) - Two similar scripts that aggregate the final word count and daily average of novels and appends results to other writer data (Basic Info, Fact Sheets)
[CalculateTextFeaturesandReadingScore](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/calc_text_features_and_reading_score.ipynb) - Classified as novel's genre as standard or nonstandard, and extracted the number of words, unique words, sentences, paragraphs, and reading score of novel synopses.
[CalculateReadingScoreExcerpts](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean/calc_flesch_reading_score-excerpts.ipynb) - Extracted the number of words, unique words, sentences, paragraphs, and reading score of novel excerpts.


### Data Dictionaries

#### Writers - About the Data

Contains basic profile information about each writer and their past NaNoWriMo statistics.

There are 501 rows and 41 columns.

The data may be found [here](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/user_summary_no2015.csv).

#### Writers - Data Dictionary
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

__FH Total__ - For the current NaNoWriMo, the total word count of a novel written in the first half of the contest

__FH Sub__ - For the current NaNoWriMo, the number of word count submissions to a novel in the first half of the contest

__SH Total__ - For the current NaNoWriMo, the total word count of a novel written in the second half of the contest

__SH Sub__ - For the current NaNoWriMo, the number of word count submissions to a novel in the second half of the contest    

#### Novels - About the Data

Contains basic profile information about each novel and their word count statistics.

There are 2122 rows and 9 columns.

The data may be found [here](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv).

#### Novels - Data Dictionary

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

#### Novel Numeric Features - About the Data

Contains numeric data represeting each novel's genre, synopses, and excerpt.

There are 2122 rows and 23 columns.

The data may be found [here](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_features.csv).

#### Novel Numeric Features - Data Dictionary
Note: There are some columns that are duplicates from the [novel_data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/clean%20data/novel_data.csv) file, so they will not be redefined here.

__has genre__ 0 if the novel has no given genre. 1 if otherwise.

__standard genre__ 1 if the novel's given genre is one of the following "usual" genres: __ . 0 if otherwise.  

__has_synopses__ 0 if the novel has no synopsis.  1 if otherwise.

__num words__ The number of words in a novel's synopsis.

__num uniques__ The number of unique words in a novel's synopsis.

__num sentences__ The number of sentences in a novel's synopsis.  

__paragraphs__ The number of sentences in a novel's synopsis.

__fk score__ The Flesch-Kincaid score of the novel synopsis.

__has excerpt__ 0 if the novel has no excerpt.  1 if otherwise.

__num words excerpt__ The number of words in a novel's excerpt.

__num uniques excerpt__ The number of unique words in a novel's excerpt.

__num sentences excerpt__ The number of sentences in a novel's excerpt.  

__paragraphs__ The number of sentences in a novel's excerpt.

__fk score excerpt__ The Flesch-Kincaid score of the novel synopsis.



## [Exploring the Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/exploratoryanalysis.ipynb)
After I had constructed the data set, I proceeded with exploring the data with Python and matplotlib visualizations.


### Exploring the Writer data

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
</div>



__Wins and Losses for NaNoWriMo 2015__

There are 219 winners and 282 nonwinners out of 501 writers.  A little over a 4:3 ratio of winners to nonwinners.  At first glance, winning is almost a coin-toss at 44%.  One has near to a 50/50 chance of guessing correctly whether or not a writer is a NaNoWriMo winner. 


__Lifetime Word Count vs Member Length__


![Imgur](http://i.imgur.com/OM5Wpyc.png)


Few writers have a written more than 1,000,000 words (or 20 NaNoWriMo winning novels) over the course of their NaNoWriMo lifetime.  The density of nonwinners for NaNoWriMo 2015 decreases as Member Length increases, and a higher Lifetime Word Count indicates higher likelihood of winning.  It makes sense that the longer one writes (Member Length) and the more words one writes (Lifetime word Count) makes one more likely to reach the NaNoWriMo writing goal.

__Expected Avg Submission vs Expected Daily Average__


![Imgur](http://i.imgur.com/VbtG1FX.png)


It almost looks like there are clusters.  If Expected Daily Average => Expected Avg Submission, a writer is more likely to win.  It's worth noting that the minimum daily average needed to win a NaNoWriMo contest is about 1,666 (50,000 words / 30 days).

__Number of Wins vs Number of times participated__


![Imgur](http://i.imgur.com/lBqjeFm.png)


It looks like there may be possible clusters here as well.  Writers who have already had more than 5 wins are very likely to win again.  Also, writers who have participated more than 5-10 times have better chances of winning as well.

__Expected Daily Average vs Expected Num Submissions__


![Imgur](http://i.imgur.com/BHBCI0G.png)


Many writers seem to cluster around an Expected Daily Average of 1500-2000.  1,666 is the minimum daily average to win a NaNoWriMo contest.  The higher an Expected Daily Average, the more likely a writer is to win the upcoming contest.  

Also interesting is how the density of nonwinners decreases as Expected Num Submissions increases, so higher Expected Num Submissions may also be indicative of winning.   


__Distribution Word Count Submissions in early weeks of a contest__

I wanted to look retrospectively at the latest NaNoWriMo contest and see how winners can be predicted as early as the first week or two weeks of a contest


![Imgur](http://i.imgur.com/TQkf7Mb.png)

![Imgur](http://i.imgur.com/LRBKpWn.png)

As expected, writers who submit more often in the early weeks are more likely to win.

__Average First Week Submissions vs Expected Daily Average__


![Imgur](http://i.imgur.com/mWKMKoy.png)


Additionally, writers whose daily average in the first week is equal to or greater than the Expected Daily Average of their past novels are more likely to win.

__Does being Municipal Liaison or having a novel sponsored have effect on winning?__


![Imgur](http://i.imgur.com/b4jSjo2.png)

Municipal Liaisons, which I've flagged with a binary variable (1 if they are a ML, 0 if otherwise), are a small fraction of the total NaNoWriMo writer population, but the majority of these MLs turn out to be winners at the end of the month.   

![Imgur](http://i.imgur.com/zMGGwTa.png)

Likewise, very few writers have sponsors for their novel.  

The ratio of winners to nonwinners for those with Sponsors is 2.  The ratio of winners to nonwinners for those who are Municipal Liaisons is almost 6.  It definitely seems like one is more likely to win if they are a Municipal Liaison of if their novel is sponsored!


### Exploring the Novel data

GET CORRECT DATA FRAME

__Overall Wins and Losses__


The total number of novels in this sample is 2123.  1333 winners and 790 nonwinners for a 63/37 split.  It's interesting that there are more winning novels than nonwinning novels while there are more winning writers for the most recent NaNoWriMo than there are nonwinning writers.  But this makes sense because writers who write more novels are more likely to have their novels reach the 50,000 word goal.

__Text Features__

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Winner</th>
      <th>Novel Date</th>
      <th>has genre</th>
      <th>standard genre</th>
      <th>has_synopses</th>
      <th>num words</th>
      <th>num uniques</th>
      <th>num sentences</th>
      <th>paragraphs</th>
      <th>fk score</th>
      <th>has excerpt</th>
      <th>num words excerpt</th>
      <th>num uniques excerpt</th>
      <th>num sentences excerpt</th>
      <th>paragraphs excerpt</th>
      <th>fk score excerpt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>November 2015</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>November 2014</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>44</td>
      <td>42</td>
      <td>3</td>
      <td>1</td>
      <td>65.73</td>
      <td>1</td>
      <td>132</td>
      <td>96</td>
      <td>13</td>
      <td>7</td>
      <td>78.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>November 2015</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>153</td>
      <td>109</td>
      <td>7</td>
      <td>4</td>
      <td>58.62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>November 2014</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>59</td>
      <td>51</td>
      <td>4</td>
      <td>3</td>
      <td>65.73</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>November 2013</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>124</td>
      <td>93</td>
      <td>4</td>
      <td>1</td>
      <td>56.93</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


![Imgur](http://i.imgur.com/WVkKWlZ.png)


The average length in words of a synopses is 50 words, or about a few good sentences.  This is likely skewed by the fact that 729 novels, more than a third, don't even have a synopsis.  There are few novels with synopses longer than 100 words, but as synopses get longer, it seems more likely that they belong to a winning novel.


![Imgur](http://i.imgur.com/sLHnooy.png)



```python
from scipy.stats import ttest_ind

ttest_ind(winlose['fk score'].get_group(0), winlose['fk score'].get_group(1))
```

    Ttest_indResult(statistic=-1.4376558464994371, pvalue=0.1506792394358735)


The Flesch-Kincaid reading scores look about normally distributed for this sample of novel synopses for both winners and nonwinners.  In a t-test comparing the two data sets, the resulting p-value is greater than 10%.  This means, I cannot reject a null hypothesis that the winning and non-winning novels have equal averages of Flesch-Kincaid scores.  Flesch-Kincaid scores for a novel synopses are unlikely to be indicative of a winning novel.  


![Imgur](http://i.imgur.com/HV69yCg.png)


Trying to plot reading score of synopses against length of synopses produces this gobbled mess.  It may be hard trying to predict winning novels with these features...




## [Logistic Regression](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/Logistic%20Regression2.ipynb)

TRY AGAIN EXCLUDING SH 

As the variable I want to predict is binary (1 if a writer is a winner, 0 if otherwise) I decided to use a logistic regression as my prediction model.  

After extracting only the numerical columns from the writer data, replacing any NaN entries - new writers who don't have data from past NaNoWriMos - with 0, I applied a Standard Scaler to normalize the data.  I then performed an 80/20 split on the data - 400 observations for training and 101 observations for testing.  

__Cross-Validation Score__

I created 10 different folds of the training data to train, test, and cross-validate a Logistic Regression model.  The average cross-validation score was  __.977__.  This is a promising indication that this model does very well in predicting a winning or non-winning outcome for a writer.  

__Confusion Matrix and Classification Report__

After cross-validating on just the training data, I re-trained the model on the entire training data set and then used the model to predict the outcomes for the writers in the test data set.  Comparing the model's predictions with the actual outcomes, I obtained the following confusion matrix and classification report.   

| | Actual 0 | Actual 1 
---|---|---
__Predicted 0__ | 51 | 4
__Predicted 1__ | 0 | 46

| | Precision | Recall | F1-Score | Support 
---|---|---|---|----
__0__ | 1.00 | 0.93 | 0.96 | 55
__1__ | 0.92 | 1.00 | 0.96 | 46
__avg/total__ | 0.96 | 0.96 | 0.96 | 101
 

Only 4 winners were misclassified as non-winners.  The Logistic Regression correctly identified the winners and nonwinners in the test data with about __96%__ accuracy, as illustrated by its precision, recall, and F1-scores.  

__ROC Curve__

In plotting the the ROC curve for the model, I found the area under the curve was almost 1.  

![Imgur](http://i.imgur.com/aMmOX0a.png)

It seems like it's a pretty good model!



### Visualize the results of the Logistic Regression with PCA


There are a lot of features in this data set, so I used Principal Components Analysis to decompose the data and easily to visualize where the winners and non-winners fall on a 2 dimensional plane.  


![Imgur](http://i.imgur.com/mogNyvc.png)


Above are the first and second principal components of the training data set, colored by the winners and nonwinners.


![Imgur](http://i.imgur.com/R1PBSCl.png)

Here's how the Logistic Regression splits the decomposed test data.  Comparing it with the actual results of the test data below, the Logistic Regression did very well generalizing the data and sorting out the winners and nonwinners of NaNoWriMo.


![Imgur](http://i.imgur.com/32VIWul.png)


Pleased with the results of the Logistic Regression model, I then similarly trained a Decision Tree on the features to compare the two methods.  


It also performed very well in predicting winners and nonwinners achieving a similar scores for cross validation and precision and recall.

| | Actual 0 | Actual 1 
---|---|---
__Predicted 0__ | 50 | 5
__Predicted 1__ | 0 | 46

| | Precision | Recall | F1-Score | Support 
---|---|---|---|----
__0__ | 1.00 | 0.91 | 0.95 | 55
__1__ | 0.90 | 1.00 | 0.95 | 46
__avg/total__ | 0.95 | 0.95 | 0.96 | 101


The Decision tree found the following features to be the most important.  


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Feature Importances</th>
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



SH Total and FH Total - the total word count of a writer's novel submitted in the second half and first half of the contest respectively - are the most predictive features of winning, but these are metrics collected after the current contest has started.  For next steps, I want to build a model with just the information I have from past contests.  


## [Using Fewer Feaures and Applying Other Models](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/Modeling%20with%20Fewer%20Features.ipynb)

I excluded the features relevant to the current contest - the number of words and submissions accounted in the first week, first two weeks, or second two weeks.  I then re-applied the Logistic Regression model.

| | Actual 0 | Actual 1 
---|---|---
__Predicted 0__ | 48 | 7
__Predicted 1__ | 23 | 23

| | Precision | Recall | F1-Score | Support 
---|---|---|---|----
__0__ | 0.68 | 0.87 | 0.76 | 55
__1__ | 0.77 | 0.50 | 0.61 | 46
__avg/total__ | 0.72 | 0.70 | 0.69 | 101

                       Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              22              24
                 precision    recall  f1-score   support
    
              0       0.69      0.87      0.77        55
              1       0.77      0.52      0.62        46
    
    avg / total       0.73      0.71      0.70       101
    
![Imgur](http://i.imgur.com/43tqjbu.png)


The difference between this model and the previous, which included the current contest data, is drastic.  Many non-winners are predicted to win... Why?  Perhaps these were past winners/active participants that just fell short this week.

I then compared the results against other models.


### Naive Bayes

TALK ABOUT CLASSIFICATION REPORT AND CONFUSION MATRIX 


    0.670064102564
    0.673267326733
    
                           Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              26              20
                 precision    recall  f1-score   support
    
              0       0.65      0.87      0.74        55
              1       0.74      0.43      0.55        46
    
    avg / total       0.69      0.67      0.65       101


Naive Bayes is not as accurate as Logistic Regression in this case.  ELABORATE probably because of it's naive assumptions

### SVM

TALK ABOUT CLASSIFICATION REPORT AND CONFUSION MATRIX 


    0.705644152595
    0.722772277228
    
                           Actual Class 0  Actual Class 1
    Predicted Class 0              49               6
    Predicted Class 1              20              26
                 precision    recall  f1-score   support
    
              0       0.71      0.89      0.79        55
              1       0.81      0.57      0.67        46
    
    avg / total       0.76      0.74      0.73       101


                      
This Support Vector Machine does a little bit better than the Logistic Regression. ELABORATE

### Decision Tree



    0.647179487179
    0.663366336634
    
                       Actual Class 0  Actual Class 1
    Predicted Class 0              42              13
    Predicted Class 1              22              24
                 precision    recall  f1-score   support
    
              0       0.66      0.76      0.71        55
              1       0.65      0.52      0.58        46
    
    avg / total       0.65      0.65      0.65       101


The Decision Tree did not do as well this time without the other features.  


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
      <td>0.359694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LifetimeWordCount</td>
      <td>0.173496</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Member Length</td>
      <td>0.085099</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Expected Num Submissions</td>
      <td>0.079384</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Expected Min Submission</td>
      <td>0.066356</td>
    </tr>
  </tbody>
</table>
</div>



Without the data from the current contest, the most important features are Expected Daily Average and LifetimeWordCount, or a writer's average daily writing productivity and how much they've participated in the past.
 

### Random Forests
I trained the data on a Random Forest using the same... The Random Forest yielded...

    0.692890869293
                     
    0.752475247525

                       Actual Class 0  Actual Class 1
    Predicted Class 0              48               7
    Predicted Class 1              18              28
                 precision    recall  f1-score   support
    
              0       0.73      0.87      0.79        55
              1       0.80      0.61      0.69        46
    
    avg / total       0.76      0.75      0.75       101


Random Forests and Support Vector Machines do best in predicting winners and nonwinners when excluding data from the current contest........

While using past data to predict the outcome of a contest is 


The activity in the first couple weeks of the contest is predictive of winning... can make or break writers who based on how well they performed in the past have good marks with potential to win 


## [Modeling Novel Data](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/NovelsModeling-Final.ipynb)
Now that I've created a model to predict which writers will be winners based on their past NaNoWriMo performances, I want to attempt to predict which novels will be winning novels based on what little I know about them: their genre, synopsis, and excerpt.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Winner</th>
      <th>Novel Date</th>
      <th>has genre</th>
      <th>standard genre</th>
      <th>has_synopses</th>
      <th>num words</th>
      <th>num uniques</th>
      <th>num sentences</th>
      <th>paragraphs</th>
      <th>fk score</th>
      <th>has excerpt</th>
      <th>num words excerpt</th>
      <th>num uniques excerpt</th>
      <th>num sentences excerpt</th>
      <th>paragraphs excerpt</th>
      <th>fk score excerpt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>November 2015</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>November 2014</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>44</td>
      <td>42</td>
      <td>3</td>
      <td>1</td>
      <td>65.73</td>
      <td>1</td>
      <td>132</td>
      <td>96</td>
      <td>13</td>
      <td>7</td>
      <td>78.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>November 2015</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>153</td>
      <td>109</td>
      <td>7</td>
      <td>4</td>
      <td>58.62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>November 2014</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>59</td>
      <td>51</td>
      <td>4</td>
      <td>3</td>
      <td>65.73</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>November 2013</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>124</td>
      <td>93</td>
      <td>4</td>
      <td>1</td>
      <td>56.93</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


Using only the above text data, DESCRIBE MODELING PROCESS (include pca).  REHIGHLIGHT TROUBLE WITH THE IMBALANCED DATA SET


__Cross-Validation Score__


__Confusion Matrix and Classification Report__


__Scoring the Test Date__


__ROC Curve__



The Logistic Regression didn't do too well this time.  SUMMARIZE RESULTS FROM THE OTHER MODELS AND WHAT MIGHT BE MOST IMPORTANT FEATURES... 


So Decision Trees and Support Vector Machines don't perform much better than guessing either.   
Maybe it just doesn't make sense to predict if a novel wins just based on it's synopses or excerpt.  Don't judge a book by it's cover I guess.  


## [Clusters of Writers](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/K%20Means.ipynb)

I've tried classifying writers by whether or not they've "won" or not in the next NaNoWriMo contest, but that sort of dampens the spirit of NaNoWriMo.  It's not just about winning after all.  I want to see what other ways to create clusters of writers with K Means.


TALK ABOUT PROCESS OF FITTING DATA TO KMEANS AND CALCULATING SILHOUETTE SCORES 

![Imgur](http://i.imgur.com/0fvoYHr.png)


It looks a k of 5 produces the best silhouette score, so the data can best be fitted into 


![Imgur](http://i.imgur.com/57jLKlo.png)


## [Genre Recommendation](https://github.com/nicaless/nanowrimo_ga_project/blob/master/analyze/Recommending%20New%20Genres.ipynb)
While I could not create a very accurate model for predicting whether or not a novel will win based on its synopses or excerpt, I still wanted to do something interesting with all the novel data I had.  So I decided to create a simple recommendation system that, given a writer's NaNoWriMo username, would suggest new genres for the writer to try writing for based on their past.  

Here's a list of...
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Writer Name</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nicaless</td>
      <td>Fantasy, Young Adult</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rachel B. Moore</td>
      <td>Literary, Literary Fiction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abookishbabe</td>
      <td>Young Adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alexabexis</td>
      <td>Romance, Horror/Supernatural, Horror &amp; Superna...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AllYellowFlowers</td>
      <td>Literary, Literary Fiction</td>
    </tr>
  </tbody>
</table>
</div>

explicit

Below defines a function that calculates the jaccard distance from two different lists of genres.


```python
def jaccard(a, b):
    if (type(a) == "float" or type(b) == float):
        return 0
    a = set(a.split(", "))
    b = set(b.split(", "))
    intersect = a.intersection(b)
    union = a.union(b)
    return float(len(intersect)) / len(union)


```
	nicaless_genres = writer_genres['Genres'][writer_genres['Writer Name'] == "Nicaless"].values[0]
	abookishbabe_genres = writer_genres['Genres'][writer_genres['Writer Name'] == "abookishbabe"].values[0]
	
	jaccard(nicaless_genres, abookishbabe_genres)

    0.5

EXPLAIN THE ABOVE SCORE

I then created a function called _getSimilar_ that uses the jaccard function to calculate the distance between a given writer's list of genres and all other writers' genres and returns a set of suggested genres based on the top ten closes writers.


```python
getSimilar("Nicaless")
```

    I suggest you try writing for the following genres:
    {'Romance', 'Science Fiction', 'Young Adult & Youth', 'nan'}
```python
getSimilar("Trillian Anderson")
```
    I suggest you try writing for the following genres:
    {'Fanfiction','Non-Fiction','Romance','Science Fiction','Steampunk','Thriller/Suspense','Young Adult','nan'}
```python
getSimilar("AmberMeyer")
```

    I suggest you try writing for the following genres:
    {'Fantasy', 'Science Fiction', 'Young Adult'}


```python
getSimilar("Brandon Sanderson")
```

    I suggest you try writing for the following genres:
    {'Romance', 'Science Fiction', 'Young Adult & Youth', 'nan'}



Cool! Looks like I have a lot in common with what Brandon Sanderson writes based on our recommendations! 

Of course, this recommender only works for writers already in my list of writers and their known past-written genres, but I'm hoping it's a list that will continue to expand.


## Conclusion

### Next Steps
feature engineering of features not based on data from current contest
expand the genre recommender
try more advanced rebalancing the novel data and then trying to fit the data
