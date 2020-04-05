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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Chelsea7227/Sentiment-Analysis/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
