#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[8]:


df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['target']=encoder.fit_transform(df['target'])


# In[11]:


df.sample(5)


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates(keep='first')


# In[15]:


df.shape


# In[16]:


df['target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[19]:


import nltk


# In[20]:


get_ipython().system('pip install nltk')


# In[21]:


nltk.download('punkt')


# In[22]:


df['num_characters'] = df['text'].apply(len)


# In[23]:


df.head()


# In[24]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[25]:


df.head()


# In[26]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[27]:


df.head()


# In[28]:


df[['num_characters','num_words','num_sentences']].describe()


# In[29]:


#non-spam
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[30]:


#spam
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[31]:


import seaborn as sns


# In[32]:


plt.figure(figsize=(10,8))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[33]:


plt.figure(figsize=(10,8))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[34]:


sns.pairplot(df,hue='target')


# In[35]:


sns.heatmap(df.corr(),annot=True)


# In[36]:


from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[37]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:];
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:];
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)


# In[38]:


nltk.download('stopwords')


# In[39]:


transform_text('Did you like my presentation on ML?')


# In[40]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[41]:


df.head()


# In[42]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[43]:


get_ipython().system('pip install wordcloud')


# In[44]:


spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[45]:


plt.imshow(spam_wc)


# In[46]:


ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[47]:


plt.imshow(ham_wc)


# In[48]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[49]:


len(spam_corpus)


# In[50]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[51]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[52]:


len(ham_corpus)


# In[53]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


df.head()


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=2000)


# In[56]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[57]:


X.shape


# In[58]:


y = df['target'].values


# In[59]:


y


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[62]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score


# In[63]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[64]:


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[65]:


mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[66]:


bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[67]:


#tfidf --> mnb


# In[68]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




