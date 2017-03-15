import re


def processTweet(tweet):
    tweet = tweet.lower()
	   
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def getFeatureVector(tweet,stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w)
    return featureVector
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
import csv

inpTweets = csv.reader(open('data/test/training_neatfile_2.csv', 'rU'),dialect=csv.excel_tab, delimiter=',', quotechar='|')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
featureList = []

tweets = []
i=0
for row in inpTweets:
    #if i>9000: break
    i+=1
    if len(row) <2 : continue
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    if i<9000:
        featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
import random
random.shuffle(tweets)
dict = list(set(featureList))
sentiment=['"positive"','"neutral"','"negative"']
print len(dict)
lX=[]
lY=[]
for tweet in tweets:
    x=[]
    for i in range(len(dict)):
        if dict[i] in tweet[0] :
            x.append(1)
        else :
            x.append(0)
    
    lX.append(x)
    lY.append(sentiment.index(tweet[1]))
    tweet=[]
import numpy as np

trainX=np.asarray(lX[:9000])
trainY=np.asarray(lY[:9000])
testX=np.asarray(lX[9000:])
testY=np.asarray(lY[9000:])
"""
inpTweets = csv.reader(open('data/training_neatfile_2.csv', 'rU'),dialect=csv.excel_tab, delimiter=',', quotechar='|')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
featureList = []
# Get tweet words
tweets = []
j=0;
for row in inpTweets:
    if j>50000 : break
    j+=1
    if len(row) <2 : continue
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    tweets.append((featureVector, sentiment));
#end loop

# Remove featureList duplicates

lX=[]
lY=[]
for tweet in tweets:
    x=[]
    for i in range(len(dict)):
        if dict[i] in tweet[0] :
            x.append(1)
        else :
            x.append(0)
    lX.append(x)
    lY.append(sentiment.index(tweet[1]))


testX=np.array(lX)
testY=np.array(lY)
"""
print trainX.shape
print trainY.shape
print testX.shape
print testY.shape
from sklearn.svm import SVC
#clf = SVC(C=100, kernel='poly',gamma=1,coef0 =1,decision_function_shape='ovo')
#clf = SVC(C=1.0, kernel='rbf',gamma=0.6,decision_function_shape='ovo')
clf = SVC(C=5.0, kernel='linear',decision_function_shape='ovo')
from datetime import datetime
print("start",str(datetime.now()))
clf.fit(trainX, trainY)
print("end train",str(datetime.now()))
from sklearn.externals import joblib
print("start dump ",str(datetime.now()))
joblib.dump(clf, 'rbf5_10000first.pkl') 
print("end dump ",str(datetime.now()))
clf = joblib.load('rbf5_10000first.pkl')
#clf = LinearSVC(C=5)
print("start predict ",str(datetime.now()))
clf.decision_function_shape = "ovr"
res=clf.predict(testX)
print("end predict ",str(datetime.now()))
sol=(res==testY)
print(res[:10])
print(testY[:10])
print(sol[:10])
print(sol.sum())
# Extract feature vector for all tweets in one shote
#training_set = nltk.classify.util.apply_features(extract_features, tweets)
