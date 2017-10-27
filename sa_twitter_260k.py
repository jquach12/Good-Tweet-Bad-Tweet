import csv
import pandas
import nltk
#from nltk.probability import *
#some re-processing
import nltk
import datetime
import string
from nltk.corpus import stopwords
from nltk.corpus import words
import random # for shuffle
#print(data)



import pickle

t0 = datetime.datetime.now()
print(t0)






print("Getting tweets and labels...")
labels = []
tweets = []

itr = 0
#read in bad tweets
with open('sentiment_tr.csv', 'r') as f:
    

    csv_reader = csv.reader(f, delimiter=',')
    
    
    for row in csv_reader:
        labels.append(row[0])
        print(row[0])
        tweets.append(row[5])




        if len(tweets) >= 131313: #for now
        	print("Read in 131313 bad tweets!")
        	break


#now get good tweets
with open('sentiment_tr.csv', 'r') as f:
    

    csv_reader = csv.reader(f, delimiter=',')
    
    
    for row in csv_reader:
    	if(row[0] != "4"):
    		continue
        labels.append(row[0])
        print(row[0])
        tweets.append(row[5])




        if len(tweets) >= 262626: #for now
        	print("Read in 131313 good tweets!")
        	break


      
print("Total tweets is: ")
print(len(tweets))
"""

for label,tweet in zip(labels,tweets):
	print (label + ": " + tweet)

"""

nltk.download('words') #get english dictionary
englishDict = set(words.words())

vocab = []

stopwords = ["i","me","my","myself","we","our","ours",
"ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her",
"hers","herself",
"it","its",
"itself","they",
"them","their",
"theirs","themselves",
'what','which',
'who','whom','this','that','these','those','am',
'is','are',
'was','were',
'be','been',
'being','have','has','had',
'having',
'do',
'does',
'did',
'doing',
'a',
'an',
'the',
'and',
'but',
'if',
'or',
'because',
'as',
'until',
'while',
'of',
'at',
'by',
'for',
'with',
'about',
'against',
'between',
'into',
'through',
'during',
'before',
'after',
'above',
'below',
'to',
'from',
'up',
'down',
'in',
'out',
'on',
'off',
'over',
'under',
'again',
'further',
'then',
'once',
'here',
'there',
'when',
'where',
'why',
'how',
'all',
'any',
'both',
'each',
'few',
'more',
'most',
'other',
'some',
'such',
'no',
'nor',
'not',
'only',
'own',
'same',
'so',
'than',
'too',
'very',
's',
't',
'can',
'will',
'just',
'don',
'should',
"now"]

print(datetime.datetime.now())
print("Getting vocab...")

#now we wanna get rid small words
for words in tweets:
	mini_vocab = []


	for w in words.split():

		if w[0] == "@" or w.startswith("www") or w.startswith("http"): # we dont care about who gets tagged
			continue

		#will refine later
		if w == "<3" or w == "</3":
			mini_vocab.append(w)
			continue

		w = w.translate(None, string.punctuation)
		w = w.lower()
		if w in stopwords or len(w) < 3:
			continue

		if w not in englishDict and len(w) > 5: #so as to allow tokens like "wth" "lmfao" "lmao" "rofl"
			continue
		
		mini_vocab.append(w) 
	vocab.append(mini_vocab)


#this gets a tuple where the first element is a list of words in the tweet and the second element is the label
labelTweet = zip(vocab,labels)





print("Just zipped up vocab and labels...")



def get_words_in_tweets(tweets):

    all_words = []

    for (words, sentiment) in labelTweet:
    	all_words.extend(words)

    return all_words

def get_word_features(wordlist):

    wordlist = nltk.FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

print("AMT OF FEATURES")
print(len(word_features))

print("Just got features...")
print(datetime.datetime.now())
#  

#for item in labelTweet:

#	print (item)

#for item in word_features:
	#print item

def get_features(document):
	d_words = set(document) #get all unique stuff
	features = {}
	for word in word_features:
		features['contains(%s)' % word ] = word in d_words
	return features


training_set = nltk.classify.apply_features(get_features,labelTweet)

print(datetime.datetime.now())
print("Now training....")
nbclf = nltk.NaiveBayesClassifier.train(training_set)


print("the time it took to train was...")
print(datetime.datetime.now())




tweetHate = "i hate you"

print("This is the prediction of your hate tweet")
print nbclf.classify(get_features(tweetHate.split()))


tweetLike = "i love you <3"

print("This is the prediction of your love tweet")
print nbclf.classify(get_features(tweetLike.split()))

#TESTING
print("Testing begins...")

print("Getting Test tweets and labels...")
testlabels = []
testtweets = []

neutralSent = 0
with open('sentiment_te.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
    	if (row[0] == "2"): #let's just omit neutral sentiment
    		print("Omitted a neutral sentiment")
    		neutralSent += 1
    		continue
        testlabels.append(row[0])
        testtweets.append(row[5])

"""
for label,tweet in zip(labels,tweets):
	print (label + ": " + tweet)

"""
print("Omitted this many neutral sentiments...")
print(neutralSent)
testvocab = []

print(datetime.datetime.now())
print("Getting vocab...")

"""
#now we wanna get rid small words
for words in tweets:
	mini_vocab = []
	for w in words.split():
		if len(w) >= 3:
			if w.isupper(): #HELLO differs than Hello
				mini_vocab.append(w)
			else:
				mini_vocab.append(w.lower())
	testvocab.append(mini_vocab)

"""

#now we wanna get rid small words
for words in testtweets:
	tmini_vocab = []


	for w in words.split():

		if w[0] == "@" or w.startswith("www") or w.startswith("http"): # we dont care about who gets tagged
			continue

		w = w.translate(None, string.punctuation)
		w = w.lower()
		if w in stopwords or len(w) < 3:
			continue

		if w not in englishDict and len(w) > 5: #so as to allow tokens like "wth" "lmfao" "lmao" "rofl"
			continue
		
		tmini_vocab.append(w) 
	testvocab.append(tmini_vocab)

#this gets a tuple where the first element is a list of words in the tweet and the second element is the label
testlabelTweet = zip(testvocab,testlabels)
print("Just zipped up the test tweets...")
print(datetime.datetime.now())

print("Testing for accuracy...")
amtRight = 0 

amtTotal = float(0)
for item in testlabelTweet:
	someString = ""
	for token in item:
		for word in token:
			someString += (word + " ")
		
		
	if nbclf.classify(get_features(someString.split())) == item[1]:
		amtRight += 1
	else:
		print (str(nbclf.classify(get_features(someString.split()))) + " DOESNT MATCH THE LABEL: " + str(item[1]) + "  TEST STR: " + someString)

	amtTotal += 1

accuracy = amtRight / amtTotal
print("AMOUNT RIGHT IS %d" %amtRight)
print("TOTAL AMOUNT OF TEST SET IS  %d" %amtTotal)

print("Accuracy: %s" %accuracy)

print("Pickling model...")

filename = 'naiveBayesSentimentAnalysis260k.sav'
pickle.dump(nbclf, open(filename, 'wb'))

print("That was fun :)")