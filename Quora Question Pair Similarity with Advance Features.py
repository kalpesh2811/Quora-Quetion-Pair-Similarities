#!/usr/bin/env python
# coding: utf-8

# ## Import  Liberies

# In[1]:


#Please install pakages first like wordcloud, re, bs4, 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup

from sklearn.metrics import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import f1_score
from wordcloud import WordCloud, STOPWORDS
from sklearn.manifold import TSNE


import warnings
warnings.filterwarnings('ignore')


# # Reading Dataset

# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


new_df = df.sample(1000,random_state=2)


# In[4]:


new_df.head()


# # Preprocesssing

# In[5]:


# function having question into lowercase and strip whitespace

def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q
    


# In[6]:


preprocess("I've already! wasn't <b>done</b>?")


# In[7]:


new_df['question1']= new_df['question1'].apply(preprocess)
new_df['question2']= new_df['question2'].apply(preprocess)


# In[8]:


new_df.head()


# # Basic Feature Engineering

# In[9]:


# Feature Engineering
# adding two more colunm into the dataframe
new_df['q1_len'] = new_df['question1'].str.len() 
new_df['q2_len'] = new_df['question2'].str.len()


# In[10]:


new_df.head()


# In[11]:


# adding another two features q1_num_words and q2_num_words
new_df['q1_num_words'] = new_df['question1'].apply(lambda row:len(row.split(" "))) 
new_df['q2_num_words'] = new_df['question2'].apply(lambda row:len(row.split(" "))) 
new_df.head()


# In[12]:


# lambda function is used for 1st lowercase and then split
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return len(w1 & w2)


# In[13]:


new_df['word_common'] = new_df.apply(common_words, axis=1)
new_df.head()


# In[14]:


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return (len(w1) + len(w2))


# In[15]:


new_df['word_total'] = new_df.apply(total_words, axis=1)
new_df.head()


# In[16]:


new_df['word_share'] = round(new_df['word_common']/new_df['word_total'],2)
new_df.head()


# ## Advance Feature Engineering

# In[17]:


#Advance Feature Engineering
from nltk.corpus import stopwords

def fetch_token_features(row): #token feature
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    token_features = [0.0]*8
    
    # Converting the sentence into tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens)==0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features
    


# In[18]:


#extract all values
import nltk
token_features = new_df.apply(fetch_token_features, axis=1)

new_df["cwc_min"]       = list(map(lambda x: x[0], token_features))
new_df["cwc_max"]       = list(map(lambda x: x[1], token_features))
new_df["csc_min"]       = list(map(lambda x: x[2], token_features))
new_df["csc_max"]       = list(map(lambda x: x[3], token_features))
new_df["ctc_min"]       = list(map(lambda x: x[4], token_features))
new_df["ctc_max"]       = list(map(lambda x: x[5], token_features))
new_df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
new_df["first_word_eq"] = list(map(lambda x: x[7], token_features))


# In[19]:


'''Sachin is a good player (token =5)(words=3)(stop words= 2)

cwc_min= ratio of no.of common word/length of small que=(3/9)=
cwc_max'''


# In[20]:


new_df.head()


# In[21]:


import distance
def fetch_length_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token mean Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features
    


# In[22]:


length_features = new_df.apply(fetch_length_features, axis=1)

new_df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
new_df['mean_len'] = list(map(lambda x: x[1], length_features))
new_df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))


# In[23]:


# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


# In[24]:


fuzzy_features = new_df.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
new_df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
new_df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
new_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
new_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


# In[25]:


print(new_df.shape)
new_df.head()


# # Graphs for Feature Engineering

# In[26]:


sns.pairplot(new_df[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')


# In[27]:


sns.pairplot(new_df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')


# In[28]:


sns.pairplot(new_df[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')


# In[29]:


sns.pairplot(new_df[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')


# In[30]:


sns.pairplot(new_df[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio','is_duplicate']],hue='is_duplicate')


# In[31]:


# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(new_df[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
y = new_df['is_duplicate'].values


# In[32]:


from sklearn.manifold import TSNE

tsne2d = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[33]:


x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=x_df, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])


# In[34]:


tsne3d = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[35]:


import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py
py.init_notebook_mode(connected=True)

trace1 = go.Scatter3d(
    x=tsne3d[:,0],
    y=tsne3d[:,1],
    z=tsne3d[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = y,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3d embedding with engineered features')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')


# # CountVectorizer

# In[36]:


#using Bag of words
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(new_df['question1']) + list(new_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2) 


# Concatnating all columns into single data frame

# In[37]:


temp_df1 = pd.DataFrame(q1_arr, index= new_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= new_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[38]:


#concatinate final dataframe with new feature
final_df = pd.concat([new_df, temp_df], axis=1)
print(final_df.shape)
final_df.head()


# # Data Analysis

# In[39]:


print("Question 1 : ")
print("\tMinimum number of characters : ", final_df['q1_len'].min())
print("\tMaximum number of characters : ", final_df['q1_len'].max())
print("\tAverage number of characters : ", final_df['q1_len'].mean())

print("Question 2 : ")
print("\tMinimum number of characters : ", final_df['q2_len'].min())
print("\tMaximum number of characters : ", final_df['q2_len'].max())
print("\tAverage number of characters : ", final_df['q2_len'].mean())

final_df.describe()


# # Intialzing X and Y

# In[40]:


#initializing x and y
# train_test_split

x = list(final_df['question1']) + list(final_df['question2'])
y = final_df['is_duplicate']
df1 = pd.DataFrame(final_df.iloc[:,8:13], index = final_df.index)
df2 = pd.DataFrame(q1_arr,index = final_df.index)
df3 = pd.DataFrame(q2_arr, index = final_df.index)
                              
xdf = pd.concat([df3,df2],axis=1)
newxdf = pd.concat([df1,xdf],axis=1)

print(newxdf.isna().sum())
print(newxdf.shape)
print(y.shape)

xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)

x=newxdf
xtrain.head()


# # Function for Confusion Matrix

# In[41]:


#Function for model Evaluation by using Confusion Metrix

def Confusion_Metrix_ME(ytest,y_pred):
    con_mat = metrics.confusion_matrix(ytest, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = con_mat, display_labels = [True,False])
    cm_display.plot()
    plt.show()
    print("Accurracy = ", accuracy_score(ytest,y_pred))
    print("Error Rate = ", 1-accuracy_score(ytest,y_pred))
    print("\n\n Classification Report : \n\n ",metrics.classification_report(ytest,y_pred))
    print("\n\n F1 - Score : ", f1_score(ytest,y_pred)) 
    print("\n Sensitivity : ",con_mat[0,0]/(con_mat[0,0]+con_mat[1,0]))   #Sensitivity (true positive rate)
    print("\n Specificity : ",con_mat[1,1]/(con_mat[1,1]+con_mat[0,1]))   #Specificity (true negative rate)
    


# # Function for ROC Curve

# In[42]:


def ROC_Curve(ytest, y_pred):
    #ROC curve is a plot of true positive rate (recall) against false positive rate (TN / (TN+FP)). AUC-ROC stands for Area Under the Receiver Operating Characteristics and the higher the area, the better the model performance. 
    #If the curve is somewhere near the 50% diagonal line, it suggests that the model randomly predicts the output variable
    print("ROC AUC Curve score : ",metrics.roc_auc_score(ytest,y_pred))
    fpr,tpr, a = metrics.roc_curve(ytest,y_pred)
    
    #ploting the ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[43]:


model_list = list()
model_train_accuracy = list()
model_test_accuracy = list()


# ## KNN Model

# In[44]:


def KNNClassifier(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.3,random_state = 42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(xtrain, ytrain)
    train_y_pred = knn.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = knn.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[45]:


model_list.append("knn")
test_accuracy,train_accuracy = KNNClassifier(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("KNN training Score : ", train_accuracy)
print("KNN Testing Score : ", test_accuracy)


# # Bernoulli Naive Bayes

# In[46]:


def BerNaiveBayes(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)
    nb = BernoulliNB(binarize = 0.0)
    nb.fit(xtrain,ytrain)
    train_y_pred = nb.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = nb.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[47]:


model_list.append("Ber_NB")
test_accuracy, train_accuracy = BerNaiveBayes(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("Bernoulli Naive Bayes training Score : ", train_accuracy)
print("Bernoulli Naive bayes Testing Score : ", test_accuracy)


# ## Guassion Naive Bayes

# In[48]:


def GaussionNaiveB(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)
    gnb = GaussianNB()
    gnb.fit(xtrain, ytrain)
    train_y_pred = gnb.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = gnb.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[49]:


model_list.append("Gau_NB")
test_accuracy,train_accuracy = GaussionNaiveB(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("Gaussion Naive Bayes training Score : ", train_accuracy)
print("Gaussion Naive bayes Testing Score : ", test_accuracy)


# ## Multinomial Naive Bayes

# In[50]:


def MultinomialNaiveB(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)
    mnb = MultinomialNB()
    mnb.fit(xtrain, ytrain)
    train_y_pred = mnb.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = mnb.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[51]:


model_list.append("Multi_NB")
test_accuracy, train_accuracy= MultinomialNaiveB(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("Multinomial Naive Bayes training Score : ", train_accuracy)
print("Multinomial Naive bayes Testing Score : ", test_accuracy)


# # Support Vector Machine Model

# In[52]:


def SupportVM(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)
    svc = SVC()
    svc.fit(xtrain, ytrain)
    train_y_pred = svc.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = svc.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[53]:


model_list.append("SVM")
train_accuracy,test_accuracy= SupportVM(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("SVM training Score : ", train_accuracy)
print("SVM Testing Score : ", test_accuracy)


# # Random Forest

# In[54]:


rf = RandomForestClassifier()
def RandomForest(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)
    rf.fit(xtrain,ytrain)
    train_y_pred = rf.predict(xtrain)
    tscore = accuracy_score(ytrain,train_y_pred)
    y_pred = rf.predict(xtest)
    score = accuracy_score(ytest,y_pred)
    Confusion_Metrix_ME(ytest,y_pred)
    ROC_Curve(ytest,y_pred)
    return score,tscore


# In[55]:


model_list.append("RandomForest")
test_accuracy, train_accuracy= RandomForest(x,y)
model_train_accuracy.append(train_accuracy)
model_test_accuracy.append(test_accuracy)
print("Random Forest training Score : ", train_accuracy)
print("Random Forest Testing Score : ", test_accuracy)


# # Model Comparision

# In[56]:


print(model_list,model_train_accuracy,model_test_accuracy)


# In[57]:


plt.bar(model_list,model_train_accuracy,  color = 'g')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Training Accuracy Comparision")
plt.show()


# In[58]:


plt.bar(model_list,model_test_accuracy,  color = 'g')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Test Accuracy Comparision")
plt.show()


# # Cross Validation

# In[59]:


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(RandomForestClassifier(),X,y,cv=10,scoring='accuracy'))


# In[60]:


'''''bias - Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model.
It always leads to high error on training and test data.

variance - variance is the variability of model prediction for a given data point or a value which tells us spread of our data. 
Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. 
As a result, such models perform very well on training data but has high error rates on test data. 

underfiting - low traing and testing accuracy 

Overfitti - high traing and low testing

pipeline
corss validation
'''


# # Testing Data

# In[61]:


testdf = pd.read_csv("test.csv")
testdf.head()
testdf.shape


# In[62]:


#preprocessing
testdf['question1']= testdf['question1'].apply(preprocess)
testdf['question2']= testdf['question2'].apply(preprocess)


# In[63]:


#Feature Engineering
testdf['q1_len'] = testdf['question1'].str.len() 
testdf['q2_len'] = testdf['question2'].str.len()


# In[64]:


# adding another two features q1_num_words and q2_num_words
testdf['q1_num_words'] = testdf['question1'].apply(lambda row:len(row.split(" "))) 
testdf['q2_num_words'] = testdf['question2'].apply(lambda row:len(row.split(" "))) 


# In[65]:


#Commen Words
testdf['word_common'] = testdf.apply(common_words, axis=1)


# In[66]:


#Total Words
testdf['word_total'] = testdf.apply(total_words, axis=1)


# In[67]:


#Words Share
testdf['word_share'] = round(testdf['word_common']/testdf['word_total'],2)
testdf.head()


# In[68]:


token_features = testdf.apply(fetch_token_features, axis=1)

testdf["cwc_min"]       = list(map(lambda x: x[0], token_features))
testdf["cwc_max"]       = list(map(lambda x: x[1], token_features))
testdf["csc_min"]       = list(map(lambda x: x[2], token_features))
testdf["csc_max"]       = list(map(lambda x: x[3], token_features))
testdf["ctc_min"]       = list(map(lambda x: x[4], token_features))
testdf["ctc_max"]       = list(map(lambda x: x[5], token_features))
testdf["last_word_eq"]  = list(map(lambda x: x[6], token_features))
testdf["first_word_eq"] = list(map(lambda x: x[7], token_features))


# In[69]:


length_features = new_df.apply(fetch_length_features, axis=1)

new_df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
new_df['mean_len'] = list(map(lambda x: x[1], length_features))
new_df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))


# In[70]:


fuzzy_features = testdf.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
testdf['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
testdf['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
testdf['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
testdf['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


# In[71]:


testdf.head()


# In[72]:


#Count vecterization
from sklearn.feature_extraction.text import CountVectorizer
test_questions = list(testdf['question1']) + list(testdf['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(test_questions).toarray(),2)
y = testdf['is_duplicate']
df1 = pd.DataFrame(testdf.iloc[:,8:], index = testdf.index)
df2 = pd.DataFrame(q1_arr,index = testdf.index)
df3 = pd.DataFrame(q2_arr, index = testdf.index)

xdf = pd.concat([df3,df2],axis=1)
newxdf = pd.concat([df1,xdf],axis=1)

print(newxdf.isna().sum())
newxdf.fillna(0)
print(newxdf.shape)
print(y.shape)
xtrain, xtest, ytrain, ytest = train_test_split(newxdf,y,test_size = 0.3,random_state = 42)

x=newxdf
print(x.shape)


# # Random Forest Classifier predicted values for unseen data

# In[73]:


rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
y_pred = rf.predict(x)
score = accuracy_score(y,y_pred)

print("Actual Values:    ",list(y))
print("Predicted Values: ",list(y_pred))
print("Accuracy Score ", score)


# In[ ]:





# In[ ]:




