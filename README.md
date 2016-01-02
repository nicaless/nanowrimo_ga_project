# Project Proposal - NaNoWriMo Winners

## Objective
Every year in November, writers all around the world participate in National Novel Writing Month (NaNoWriMo) and try to write 50,000 words of a novel within 30 days.  They track their word count progress on the NaNoWriMo website where they may also donate to the writing cause, join 'Regions' for writing camaraderie, and display the summary of their novel in progress.  Those writers who write 50,000 words before the end of November are declared 'Winners'. 

I plan to create a machine learning model that can predict whether a writer will be a NaNoWriMo 'winner' using data from the site.  

## The Data

NaNoWriMo has a web API that can be used to extract the following data:

**User Data**

- user name
- word count submitted over time
- whether or not they are a 'Winner'

**Region Data**

- region name
- number of participants
- Average, StDev, Min, Max, Word Count Submissions from users in that region over time
- number of donors in that region over time
- Total donation amount over time

In reaching out to the NaNoWriMo organizers, I hope to also acquire the following _**additional data**_:

**User Data**

- What region(s) they belong to
- Whether or not they are a donor
- Past NaNoWriMos they have participated in and whether or not they were a winner

**Region Data**

- Average, StDev, Min, Max, Word Count Submissions over time for past NaNoWriMos

## Challenges

I already have experience extracting the data using the web API and parsing the XML structure for word count related data.  I have written functions using R but to streamline the process I will likely rewrite the functions in Python.  This should not be too difficult using Pandas and Python's XML modules.  

I am unsure of the structure of the additional data, so processing that data and connecting it to the data I currently have is a potential challenge.  

Another challenge is the event I am not able to receive the additional data.  Much of this data is still accessible via the website, but I would need to manually scrape for the information I need. 

In the worse case, I will only be able to work with the word count data I currently have, so creating an accurate model with little data is another huge challenge.  

## Motivation

I love writing and I am fascinated by NaNoWriMo.  This idea stems from another personal project to create my own [Word Count Tracker](http://nicaless.github.io/2015/11/09/My%20First%20Shiny%20App.html).  It started off as simply an exploratory endeavor and a way to practive visualizations in R and become familiar with R's Shiny package for interactive visualization applications.  It will be incredibly fun to recreate and extend this project using Python.  I hope to find new insights in the data by creating this predictive model.  I also hope such a model may be able to help other writers and future participants in NaNoWriMo improve their writing strategies and become motivated to continue to write.   