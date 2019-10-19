#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['retina']")
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import os
# os.listdir("../input")
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


df=pd.read_csv('GrammarandProductReviews.csv')
df.head()


# In[3]:


print ('Rows     : ', df.shape[0])
print ('Columns  : ', df.shape[1])
print ('\nFeatures : ', df.columns.tolist())
print ('\nMissing values :  ', df.isnull().sum().values.sum())
print ('\nUnique values :  \n', df.nunique())


# In[4]:


df.info()
df.isnull().sum()


# In[5]:


df = df.dropna(subset=['reviews.text'])


# In[6]:


df['reviews.text'].value_counts()[10:50]


# In[7]:


plt.figure(figsize = [10,5])

x = list(df['reviews.rating'].value_counts().index)
y = list(df['reviews.rating'].value_counts() )
plt.barh(x, y)
ticks_x = np.linspace(0, 50000, 6)
plt.xticks(ticks_x, fontsize = 10, family = 'fantasy', color = 'black')
plt.yticks(size = 15)
plt.title('Distribution of Ratings', fontsize = 20, weight = 'bold', color = 'navy', loc = 'center')
plt.xlabel('Count', fontsize= 15, weight = 'bold', color ='navy')
plt.ylabel('Ratings', fontsize=15, weight = 'bold', color = 'navy')
plt.legend(['Review Rating'], shadow = True, loc= 4)


# In[8]:


from wordcloud import WordCloud, STOPWORDS


# In[9]:


stopwords = set(STOPWORDS)

def show_wordcloud(data, title =None):
    wordcloud = WordCloud(background_color = 'white', stopwords = stopwords, max_words = 300, max_font_size = 40, scale = 3, random_state = 1).generate(str(data))
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    
    if title:
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)
    
    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['reviews.text'])


# In[10]:


wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=300, max_font_size=40, scale=3,
        random_state=1).generate(str(df['reviews.text'].value_counts()))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[11]:


show_wordcloud(df['reviews.title'])


# In[12]:


wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=400, max_font_size=40, scale=30,
        random_state=1).generate_from_frequencies((df['reviews.title'].value_counts()  ))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[13]:


split_title = []
listCounts = []
split_title = [x.split(" ") for x in df['reviews.title'].astype(str)]
big_list = []
for x in split_title:
    big_list.extend(x)

listCounts = pd.Series(big_list).value_counts()
wordcloud = WordCloud(background_color='white', max_words=400, max_font_size=40, scale=30, random_state=1).generate((listCounts[listCounts > 2]).to_string())
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[14]:


len(big_list)


# In[15]:


split_title = []
listCounts = []
split_title = [x.split(" ") for x in df['reviews.text'].astype(str)]
big_list = []
for x in split_title:
    big_list.extend(x)

listCounts = pd.Series(big_list).value_counts()

wordcloud = WordCloud(background_color='white', max_words=400, max_font_size=40, scale=30,
        random_state=1).generate((listCounts[listCounts > 2]).to_string())
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[16]:


len(big_list)   # reviews.text


# In[17]:


df['categories'].value_counts()


# In[18]:


df['name'].value_counts()


# In[19]:


df['reviews.didPurchase'].fillna('Null',inplace = True)


# In[20]:


plt.figure(figsize=(10,8))
ax = sns.countplot(df['reviews.didPurchase'])
ax.set_xlabel(xlabel="Shoppers did purchase the product", fontsize=17)
ax.set_ylabel(ylabel='Count of Reviews', fontsize=17)
ax.axes.set_title('Number of Genuine Reviews', fontsize=17)
ax.tick_params(labelsize=13)


# In[21]:


df['reviews.didPurchase'].value_counts()


# In[22]:


sns.set(font_scale = 1.4)
plt.figure(figsize = (10,5))
sns.heatmap(df.corr(), cmap='coolwarm', annot = True, linewidth =.5)


# In[23]:


df1 = df[df['reviews.didPurchase']==True]
df1['name'].value_counts()


# In[24]:


df1['name'].value_counts()[0:10].plot('barh', figsize=[10,6], fontsize=20).invert_yaxis()


# In[25]:


df1 = df1[df1['name'] == 'The Foodsaver174 10 Cup Fresh Container - Fac10-000']
df1 = df1[df1['reviews.rating']==5]
# keep relevant columns only
df1 = df1[[ 'reviews.rating', 'reviews.text']]
df1


# In[31]:


df1.to_csv('Foodsaver_5.csv', index = False)


# In[32]:


from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# In[33]:


all_text = df['reviews.text']
y = df['reviews.rating']


# In[34]:


word_vectorizer = TfidfVectorizer(sublinear_tf = True, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}',
                                 stop_words = 'english', ngram_range = (1,3) )

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(all_text)


# In[35]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(all_text)

train_features = hstack([train_char_features, train_word_features])


# In[36]:


import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
seed = 71


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.3, random_state=seed)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
time1 = time.time()
classifier = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = seed, n_jobs = -1)
classifier.fit(X_train, y_train)
preds1= classifier.predict(X_test)
time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


print(accuracy_score(preds1, y_test))
print(classification_report(preds1, y_test))
print(confusion_matrix(preds1, y_test))


# In[ ]:


import xgboost as xgb
time1 = time.time()

xgb = xgb.XGBClassifier(n_jobs=-1)
xgb.fit(X_train,y_train)
preds2=xgb.predict(X_test)
time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# manual method to check accuracy, see first 100 predictions, around 70% correct prediction
for i in range(100):
    if preds2[i] == np.array(y_test)[i]:
        print('1', end=', ')   # correct prediction
    else:
        print('0', end=', ')   # wrong prediction


# In[ ]:


preds2[0:100: 5]


# In[ ]:


np.array(y_test)[0:100: 5]


# In[ ]:



correct = 0
wrong = 0
for i in range(21303):
    if preds2[i] == np.array(y_test)[i]:
        correct += 1
    else:
        wrong += 1
print(correct+wrong)
print(correct/21303)


# In[ ]:


print("XGBoost Model accuracy", accuracy_score(preds2, y_test))
print(classification_report(preds2, y_test))
print(confusion_matrix(preds2, y_test))


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


# In[39]:


time1 = time.time()

logit = LogisticRegression(C=1, multi_class='ovr')
logit.fit(X_train,y_train)
preds3 = logit.predict(X_test)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[40]:


print("Logistic Regression accuracy", accuracy_score(preds3, y_test))
print(classification_report(preds3, y_test))
print(confusion_matrix(preds3, y_test))


# In[41]:


df['sentiment'] = df['reviews.rating']<4


# In[42]:


from sklearn.model_selection import train_test_split
train_text, test_text, train_y, test_y = train_test_split(df['reviews.text'],df['sentiment'],test_size = 0.2)


# In[43]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


# In[44]:


MAX_NB_WORDS = 20000

texts_train = train_text.astype(str)
texts_test = test_text.astype(str)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level = False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[45]:


MAX_SEQUENCE_LENGTH = 200

x_train = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen = MAX_SEQUENCE_LENGTH)
print(x_train.shape)
rint(x_test.shape)


# In[46]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2,input_shape = (1,)))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

model.fit(x_train, train_y,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, test_y))

