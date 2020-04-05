## Welcome to GitHub Pages of Group Point5

In the course of text analytics and natural language processing in finance and fintech，our group try to analyze “Does sentiment affect the volatility Index”, below are things we want to share about the  project.

## Blog 1
First, we downloaded data needed.
We downloaded tweet from twitter ,with text_query #coronavirus,stock”
(when we downloaded data using this code, we met problems like missing data of some days. We cannot figure out what is going wrong so we change to download the data day by day, but that code is too long and repetitive, so we still present the initial code)

```markdown

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

```

After data downloading, we started our analysis. At the beginning we tried to use textblob to score the tweet content and obtain sentiment value.And textblob. Polarity has a value [-1,0] represent negative sentiment and [0,1] represent positive sentiment.
Then we cleaned the data, we removed URL,non-English language, mentions and all the other things we thought useless in the sentences.

```markdown

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

```markdown
And calculated average polarity every day. 

# get daily sentiment
def get_date(text):
    s = text[0:10]
    return s
df=train.groupby("date").mean().rename(columns={"polarity":"polarity_mean","subjectivity":"subjectivity_mean"})
df

**Bold** and _Italic_ and `Code` text

```

However, we found that all the average polarity were positive. We thought it may be caused by the inaccuracy of textblob. And we cannot use this data of sentiment to predict the volatility index.
Thus, we tried to build a sentiment analysis model to help us classify the emotions in tweet better. And we use the dataset sentiment140 to train our model and have an accuracy closed to 80%.

## Blog 3
After trying the linear regression model and logit model, we decided to try another model called random forest model. why we choose it? To answer this question, we will suggest some of its advantages which will clear your mind. The first advantage is that it can both handle binary features and numerical features. Secondly, it is great with high dimensionality. Thirdly, it is robust to non-linear data.

Our initial idea is to compare the importance of sentiment to other parameters, so we think the cross_val_score function will be very suitable for us to achieve our goal. So we set up three tests, the first one is to do the cross validation for each feature. Then we calculate the overall score with all features and drop the key parameter—sentiment to get a new one. In this way, we can compare the importance of different features. 

```markdown

rf = RandomForestClassifier(n_estimators=20, max_depth=4)
scores = []
# cross-validation to evaluate each model
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), 0.3))
    scores.append((float(format(np.mean(score), '.3f')), names[i]))

# with all feature
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X), 0.3))
scores.append((float(format(np.mean(score), '.3f')), 'all features'))

# drop ‘Sentiment‘ feature
X = np.delete(X, 0, axis=1)
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X),0.3))
scores.append((float(format(np.mean(score), '.3f')), 'drop Sentiment feature'))

```

A problem we encounter during the process is that, at first, we try to set “n_estimators” as 40, because, in general, the more trees in the forest, the more robust would be the prediction and thus higher accuracy, but it takes a long time for the code to run. To save time, we finally decided to cut it a half. There is always a trade-off between efficiency and accuracy.

Complete Python code is listed as follows:

```markdown

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt 

head = []
X = []
Y = []
with open('raw_data', 'r') as f:
    for line in f:
        lis = line.strip(' \n').split('\t')
        if len(head)==0:
                head = lis
        else:
            con = []
            for i in range(1,len(head)):
                x = lis[i]
                if x=='':
                    con.append(0.0)
                else:
                    if '%' in x:
                        x = float(x.replace('%',''))/100.
                    else:
                        x = float(x)
                    con.append((x))
            X.append(con)
            Y.append(int(lis[0]))

names = head[1:len(head)]
X = np.array(X)
Y = np.array(Y)
print names
print X
print Y

rf = RandomForestClassifier(n_estimators=20, max_depth=4)
scores = []
# cross-validation to evaluate each model
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), 0.3))
    scores.append((float(format(np.mean(score), '.3f')), names[i]))

# with all feature
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X), 0.3))
scores.append((float(format(np.mean(score), '.3f')), 'all features'))

# drop ‘Sentiment‘ feature
X = np.delete(X, 0, axis=1)
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X),0.3))
scores.append((float(format(np.mean(score), '.3f')), 'drop Sentiment feature'))


print(sorted(scores, reverse=True))

ylabel=map(itemgetter(0), scores)
label=map(itemgetter(1), scores)
plt.bar(range(len(label)), ylabel,tick_label=label)
plt.tick_params(labelsize=8)

i=0
for a, b in zip(range(len(label)), ylabel):
    if i<=len(label):
        plt.text(a, b, ylabel[i], ha='center', va='bottom', fontsize=11)
        i=i+1
plt.show
```
