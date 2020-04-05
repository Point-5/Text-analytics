## Welcome to GitHub Pages of Group Point5

In the course of text analytics and natural language processing in finance and fintech，our group try to analyze “Does sentiment affect the volatility Index”, below are things we want to share about the  project.

First, we downloaded data needed.
We downloaded tweet from twitter ,with text_query #coronavirus,stock”
(when we downloaded data using this code, we met problems like missing data of some days. We cannot figure out what is going wrong so we change to download the data day by day, but that code is too long and repetitive, so we still present the initial code)

```markdown
Syntax highlighted code block

import GetOldTweets3 as got
import csv
# Creation of query_search
text_query = '#coronavirus,stock'
since_date = '2020-01-21'
until_date = '2020-03-25'
count = 10000
# Creation of query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_date).setUntil(until_date).setMaxTweets(count)
# Creation of list that contains all tweets
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
# Creating list of chosen tweet data
text_tweets = [[tweet.date, tweet.text] for tweet in tweets]
# Save tweets we downloaded 
outfile = open('0121_0325_try.csv', 'w', newline = '')
writer = csv.writer(outfile)
writer.writerow(['datetime','content'])
for tweet in text_tweets:
    writer.writerow(tweet)
outfile.close()
import pandas as pd
import os
# Input the path 
Folder_Path = ''         
SaveFile_Path = ''
SaveFile_Name = 'data_new.csv'             
# Save data in the form of dataframe
os.chdir(Folder_Path)
file_list = os.listdir()
df = pd.read_csv(Folder_Path +'/'+ file_list[0])   
df.to_csv(SaveFile_Path+'/'+ SaveFile_Name,encoding="utf_8_sig",index=False)
for i in range(1,len(file_list)):
    df = pd.read_csv(Folder_Path + '/'+ file_list[i])
    df.to_csv(SaveFile_Path+'/'+ SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')
# Sort tweets and create a new file-sorted_new.csv
import sys, csv , operator
data = csv.reader(open('data_new.csv'),delimiter=',')
sortedlist = sorted(data, key = lambda x: x[0],reverse=False)
print(sortedlist)
with open("sorted_new.csv", "w", newline = '') as f:
    fileWriter = csv.writer(f, delimiter=',')
    fileWriter.writerow(['datetime','content'])
    for row in sortedlist[:len(sortedlist)-1]:
            print(row)
            fileWriter.writerow(row)
f.close()

**Bold** and _Italic_ and `Code` text

```

After data downloading, we started our analysis. At the beginning we tried to use textblob to score the tweet content and obtain sentiment value.And textblob. Polarity has a value [-1,0] represent negative sentiment and [0,1] represent positive sentiment.
Then we cleaned the data, we removed URL,non-English language, mentions and all the other things we thought useless in the sentences.

```markdown
Syntax highlighted code block

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from textblob import TextBlob
warnings.filterwarnings("ignore", category=DeprecationWarning)

train  = pd.read_csv('sorted.csv')
# remove url
def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

train.content = train.content.apply(remove_urls)
# remove non-en language
train = train[~train['content'].str.contains(r'[^\x00-\x7F]+')]
train.head()
# remove pattern
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt    
# remove twitter handles (@user)
train['tidy_tweet'] = np.vectorize(remove_pattern)(train['content'], "@[\w]*")
# remove special characters, numbers, punctuations
train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# romove short words
train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# take a look
train.head()


```
Then we scored all the text that we obtained from twitter.

```markdown
Syntax highlighted code block
# score it
text = train.tidy_tweet.iloc[0]
s = TextBlob(text)
print(s)
def get_polarity(text):
    s = TextBlob(text)
    return s.sentiment.polarity
print(s.sentiment)
def get_subjectivity(text):
    s = TextBlob(text)
    return s.sentiment.subjectivity
# get the polarity and subjectivity of tweets
train["polarity"] = train.tidy_tweet.apply(get_polarity)
train["subjectivity"] = train.tidy_tweet.apply(get_subjectivity)
train.head()
# get daily sentiment
def get_date(text):
    s = text[0:10]
    return s
df=train.groupby("date").mean().rename(columns={"polarity":"polarity_mean","subjectivity":"subjectivity_mean"})
df

**Bold** and _Italic_ and `Code` text

```
