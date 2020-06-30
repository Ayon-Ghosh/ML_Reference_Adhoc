# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:09:02 2019

@author: 140524
"""
#CONTENT BASED RECOMMENDATION ENGINE
# https://github.com/codeheroku/Introduction-to-Machine-Learning/blob/master/Building%20a%20Movie%20Recommendation%20Engine/movie_recommender_completed.py
#https://www.youtube.com/watch?v=3ecNC-So0r4&list=PLYU7hR8tUkqRWCeTUjHDHfYwGGOQXHGHt&index=25

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##Step 1: Read CSV File
df = pd.read_csv('https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv')
df.shape
#df.head()
df.columns
df.isnull().sum()
#df.title.head(100)

##Step 2: Select Features

features = ['keywords','cast','genres','director']
df[features].head()
for feature in features:
    df[feature] = df[feature].fillna('')
    


## combine features into a single column

def combine_row(row):
     return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
 

df["combined_features"] = df[features].apply(combine_row, axis=1)   
df["combined_features"].head(0)
df["combined_features"].shape 

##Step 4: Create count matrix from this new combined column 
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
count_matrix.shape
# looking at it in array for
print(count_matrix)
count_matrix.toarray()
count_matrix.toarray().shape

##Step 5: Compute the Cosine distance/Similarity and fill up the count_matrix
cosine_sim = cosine_similarity(count_matrix) 
cosine_sim[0]
 

# looking at the matrix in DataFrame form bcoz it is easier to view
df_cosine = pd.DataFrame(cosine_sim)
df_cosine.head()

##Step 5: Compute the Cosine Similarity of movie_user_likes = "Avatar"

# Get the index of Avatar

#index_Av = df[df['original_title']=='Avatar'].index.values[0]
#index_Av

#get index from movies
def get_index_from_title(movie_name):
                 return  df[df.title==movie_name]['index'].values[0]

# get title from index             
             
def get_title_from_index(index):
                 return  df[df.index==index]['title'].values[0]

# find the recommendations for the couple of movies
                 
movie_name = 'Avatar'
#movie_name = 'Inception'
movie_index = get_index_from_title(movie_name)
movie_index
# in order to enumerate we have convert cosine into list
list_cosine_sim = list(cosine_sim[0])
list_cosine_sim
recomm_list = list(enumerate(list_cosine_sim))
recomm_list
# findind the cosine similarity of the movie
# method 1
def second(x):
    return x[1]
recomm_list.sort(key=second,reverse=True)     
recomm_list        
#print(sorted_reco)                 
# or

sorted_reco = sorted(recomm_list, key = lambda x: x[1], reverse = True)
sorted_reco
i=1
for mov in sorted_reco:
    print(get_title_from_index(mov[0]))
    i = i+1
    if i>50:
        break

#USER TO USER COLLABORATIVE FILTERING FOR MOVIE  

# ITEM TO ITEM COLLABORATIVE FILTERING         
        
    
    