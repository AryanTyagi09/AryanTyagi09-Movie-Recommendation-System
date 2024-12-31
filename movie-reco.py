
"""Description - Here we are using tmdb Dataset and our focus to recommend movie to user on the basis of content based filtering
Dataset link - https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"""
import pandas as pd
import numpy as np
import pickle
import gdown
import sklearn

credit_file_id = "1FVl7aQf5DlZrh5A3_-fhDBFvydAS_7FY"
movies_file_id = "1-f2kCgUZJM08Q42HNQHU_tsCkBqXtWG9"

# Download files
credit_url = f"https://drive.google.com/uc?id={credit_file_id}"
movies_url = f"https://drive.google.com/uc?id={movies_file_id}"

credit_file_path = "tmdb_5000_credits.csv"
movies_file_path = "tmdb_5000_movies.csv"

gdown.download(credit_url, credit_file_path, quiet=False)
gdown.download(movies_url, movies_file_path, quiet=False)

credit = pd.read_csv(credit_file_path)
movies = pd.read_csv(movies_file_path)


#credit = pd.read_csv('C:/Users/HP/Desktop/Project/tmdb_5000_credits.csv')
#movies = pd.read_csv('C:/Users/HP/Desktop/Project/tmdb_5000_movies.csv')



# /content/drive/MyDrive/Campus x/End to End Project /Movie Recomended system /tmdb_5000_movies.csv
# /content/drive/MyDrive/Campus x/End to End Project /Movie Recomended system /tmdb_5000_credits.csv

movies.head(1)
credit.head(1)
credit.head(1)['crew'].values

movies = movies.merge(credit,on='title')

movies.head(1)
# Best feature selection -
#movie_id, keyword, title ,overview,cast ,crew,genres ,cast

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.info()
movies.head(1)
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres
# we will change the format of data
 # ['Action','Adventure','FFantasy','SciFi']
def convert(obj):
  L =[]
  # we will create a function and make a loop on list
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L
    # from a loop we will get a dictionary ,and we will append the data in list L

import ast  # This libarary we will add in function to make output in dic form .
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
#convert([{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}])

# now we will apply function , after add ast in function
movies['genres'].apply(convert)

movies['genres']= movies['genres'].apply(convert)
movies.head()

# now we will do samething for keyword
movies['keywords']=movies['keywords'].apply(convert)
movies.head()

movies['cast'][0]
# now we will make function for cast and substract the cast column
def convert3(obj):
  L =[]
  counter = 0
  # we will create a function and make a loop on list
  for i in ast.literal_eval(obj):
    if counter!= 3:
       L.append(i['name'])
       counter+=1
    else:
        break
  return L


movies['cast']=movies['cast'].apply(convert3)


movies.head()

movies['crew'][0]
def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i['name'])
      break
  return L


movies['crew']=movies['crew'].apply(fetch_director)

movies.info()

# now overview col is string ,we will convert this in list

movies['overview'][0]

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

# now we will add all list and convert again in list and then convert in string
# and we will get a paragraph and which is our tag column

# now we have a problm - we need to apply a tranformer , if there is any space in col value it will remove
# Sam Worthington =SamWorthington , we will do like that , it is easy for our recomended system ,
# Beacase if Sam is on another row it will confuse our model

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
# to remove space from string
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

movies.head(1)

# now we will creata a col tag
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['cast']

movies.head(1)

new_df = movies.drop(columns=['overview','genres','keywords','cast','crew'])

new_df
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# we are converting list in to string
new_df.head()

new_df['tags'][0]

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

new_df.head(1)

# now we will focus on vectorization for text
# Our focus to calculate similiarity between all text to find out the relation between them .
# We will use bag of words approach - we will extract most 500 frequent words form tag column and we will remove stopwords and apply vectorization method .


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y=[]
  for i in text.split(): # string ko list me convert kr denge
    y.append(ps.stem(i))
    # now we are converting it again in list
  return " ".join(y)




new_df['tags']=new_df['tags'].apply(stem)

# now our task- our every movie is in vector in row form in data , so our  focus is to calculate distance between vector
# here we will calculate the consine distance instead of eculidiean distance and it will work on angle value - jitna jayada angle honga value utni hi badi hongi


#cv.fit_transform(new_df['tags']).toarray().shape
# we will convert in  the numpy array and will see size and this is the vector form

vectors=cv.fit_transform(new_df['tags']).toarray()
vectors


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity


similarity.shape

similarity[0]
# it is the output of 1st array and the similarity of first movie with another movies, and digonal with first movie will be one.



def recommend(movie):
  movie_index = new_df[new_df['title']==movie].index[0]
  distances =similarity[movie_index]               #calculating the distance and doing sort them
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(new_df.iloc[i[0]].title)


  #logic explain - now our target is to make a function recomended - if i will give movie name it will give me 5 movie name
  # we are taking enumerate function - we will take index of movie and will take the index of movie and will take  similarity matrix and will sort disntacne to calculate similar matrix.

  recommend('Avatar')


new_df[new_df['title']=='Batman Begins']

new_df[new_df['title']=='Batman Begins'].index[0]  # we getting the index
# we want sorting in ascending order
sorted(similarity[0],reverse=True)[:5]
# Here we will loosing our index so for this we use enumerate function.


similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])   # here we have created index and will apply lambda to take simmiliarity of our index
# Here is the sorting on the basis of second vector .

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))    # dump the file in dictionary form , getting some error in streamlit during making website.

#import pickle
#pickle.dump(new_df,open('movies.pkl','wb'))  - Not working showing error

pickle.dump(similarity,open('similarity.pkl','wb'))
