import numpy as np 
import pandas as pd 
import nltk 
import string
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pickle
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
os.chdir("C:\\Users\\user1\\Desktop\\Final year project\\train")
df = pd.read_csv('train.csv')
lemmatiser = WordNetLemmatizer()
def text_process(tex): 
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct) 
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    
    return [word for word in a.split() if word.lower() not 
            in stopwords.words('english')]
def unique_words(words):
    word_count = len(words)
    unique_count = len(set(words))
    return unique_count / word_count

y = df['author']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
X = df['text']
wordcloud1 = WordCloud().generate(X[0]) # for EAP
wordcloud2 = WordCloud().generate(X[1]) # for HPL
wordcloud3 = WordCloud().generate(X[3]) # for MWS 
print(X[0])
print(df['author'][0])
plt.imshow(wordcloud1, interpolation='bilinear')
plt.savefig('wordcloud1.png')
plt.show()
print(X[1])
print(df['author'][1])
plt.imshow(wordcloud2, interpolation='bilinear')
plt.savefig('wordcloud2.png')
plt.show()
print(X[3])
print(df['author'][3])
plt.imshow(wordcloud3, interpolation='bilinear')
plt.savefig('wordcloud3.png')
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1234)

bow_transformer = pickle.load(open('nlp_bow_transformer.sav', 'rb'))
text_bow_train=bow_transformer.transform(X_train)
text_bow_test=bow_transformer.transform(X_test)
model = pickle.load(open('nlp_model.sav', 'rb'))
trainval=(model.score(text_bow_train, y_train))
testval=(model.score(text_bow_test, y_test))
f = open("accuracy.txt", "a")
f.write("Train data Accuracy = {}\nModel Accuracy = {}".format(trainval,testval))
f.close()



from sklearn.metrics import classification_report
predictions = model.predict(text_bow_test)
classification_rept=''+(classification_report(y_test,predictions))
f = open("classification_rept.txt", "a")
f.write(classification_rept)
f.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
cm = confusion_matrix(y_test,predictions)
plt.figure()
plot_confusion_matrix(cm, classes=['EAP','HPL','MWS'], normalize=True,title='Confusion Matrix')



os.chdir("C:\\Users\\user1\\Desktop\\Final year project\\train")
df = pd.read_csv('train.csv')
p=str(df.head())
f = open("dataset_head.txt", "a")
f.write(p)
f.close()

no_punct_translator=str.maketrans('','',string.punctuation)
df['words'] = df['text'].apply(lambda t: nltk.word_tokenize(t.translate(no_punct_translator).lower()))

df['word_count'] = df['words'].apply(lambda words: len(words))
df['sentence_length'] = df['words'].apply(lambda w: sum(map(len, w)))
df['text_length'] = df['text'].apply(lambda t: len(t))

p=str(df.head())
f = open("dataset_head2.txt", "a")
f.write(p)
f.close()


sns.set_style('whitegrid')


boxplott=sns.boxplot(x = "author", y = "word_count", data=df, color = "red")


df['punctuation_count'] = df['text'].apply(lambda t: len(list(filter(lambda c: c in t, string.punctuation))))

df['punctuation_per_char'] = df['punctuation_count'] / df['text_length']

df['unique_ratio'] = df['words'].apply(unique_words)
df.groupby(['author'])['unique_ratio'].describe()
authors = ['MWS', 'HPL', 'EAP']

for author in authors:
    sns.distplot(df[df['author'] == author]['unique_ratio'], label = author, hist=False)

plt.legend();
plt.savefig('unique_ratio.png')


avg_length = lambda words: sum(map(len, words)) / len(words)

df['avg_word_length'] = df['words'].apply(avg_length)

for author in authors:
    sns.distplot(df[df['author'] == author]['avg_word_length'], label = author, hist=False)
plt.legend();
plt.savefig('average_word_length.png')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

df['sentiment'] = df['text'].apply(lambda t: sid.polarity_scores(t)['compound'])
for author in authors:
    sns.distplot(df[df['author'] == author]['sentiment'], label = author, hist=False)

plt.legend();
plt.savefig('sentiment.png')