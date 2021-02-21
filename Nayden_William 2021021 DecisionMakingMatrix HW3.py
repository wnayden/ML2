# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:13:03 2021

@author: WilliamNayden
"""
#%%
# Decision Making With Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch. You have to pick a restaurant that’s best for everyone.  Then you should decided if you should split into two groups so everyone is happier.  

# Despite the simplicity of the process you will need to make decisions regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import rankdata
import math
import scipy.stats as ss

#%% Functions
def getRank(data):
	return(ss.rankdata(len(data)-ss.rankdata(data)+2,method='min'))
#%%
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
people = {'Jane': {'willingness to travel': 5
                  ,'desire for new experience': 3
                  ,'cost': 8 
                  #,'indian food': 8 
                  #,'mexican food': 9
                  #,'hipster points': 1
                  ,'vegetarian':  5
                  }
          ,'Max': {'willingness to travel': 9
                  ,'desire for new experience': 6
                  ,'cost': 9 
                  #,'indian food': 3
                  #,'mexican food': 4
                  #,'hipster points': 1
                  ,'vegetarian':  1
                  }
		   ,'Anna Maria': {'willingness to travel': 9
                  ,'desire for new experience': 8
                  ,'cost': 4 
                  #,'indian food': 5
                  #,'mexican food': 6
                  #,'hipster points': 2
                  ,'vegetarian':  1
                  }
		     ,'Letizia': {'willingness to travel': 9
                  ,'desire for new experience': 8
                  ,'cost': 2 
                  #,'indian food': 8
                  #,'mexican food': 3
                  #,'hipster points': 5
                  ,'vegetarian':  9
                  }
			  ,'Daniele': {'willingness to travel': 6
                  ,'desire for new experience': 5
                  ,'cost': 7 
                  #,'indian food': 5
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegetarian':  5
                  }
			  ,'Brooke': {'willingness to travel': 3
                  ,'desire for new experience': 3
                  ,'cost': 4 
                  #,'indian food': 9
                  #,'mexican food': 3
                  #,'hipster points': 7
                  ,'vegetarian':  8
                  }
			  ,'David': {'willingness to travel': 5
                  ,'desire for new experience': 3
                  ,'cost': 6
                  #,'indian food': 4
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegetarian':  5
                  }
			  ,'Joe': {'willingness to travel': 9
                  ,'desire for new experience': 7
                  ,'cost': 1
                  #,'indian food': 8
                  #,'mexican food': 5
                  #,'hipster points': 1
                  ,'vegetarian':  5
                  }
			  ,'Diana': {'willingness to travel': 3
                  ,'desire for new experience': 2
                  ,'cost': 7
                  #,'indian food': 2
                  #,'mexican food': 5
                  #,'hipster points': 4
                  ,'vegetarian':  8
                  }
			  ,'Jeremy': {'willingness to travel': 5
                  ,'desire for new experience': 2
                  ,'cost': 2
                  #,'indian food': 6
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegetarian':  2
                  }
          }         

#%%
# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
people_names = list(people)
people_cols =list(people[people_names[1]])

M_people = np.zeros((len(people_names),len(people_cols)))
for i, p in enumerate(people):
	M_people[i,] = np.array(list(people[p].values()))

print(M_people)
#%%
# Next you collected data from an internet website. You got the following information.

restaurants  = {'flacos':{'distance' : 3
                        ,'novelty' : 2
                        ,'cost': 1
                        #,'average rating': 5
                        #,'cuisine': 8
                        ,'vegetarians': 3
                        }
				  ,'Pizza Hut':{'distance' : 9
                        ,'novelty' : 1
                        ,'cost': 9
                        #,'average rating': 2
                        #,'cuisine': 2
                        ,'vegetarians': 4
                        }
				  ,'Flat Bread':{'distance' : 8
                        ,'novelty' : 4
                        ,'cost': 4
                        #,'average rating': 7
                        #,'cuisine': 8
                        ,'vegetarians': 6
                        }
				  ,'10 Barrels':{'distance' : 6
                        ,'novelty' : 6
                        ,'cost': 5
                        #,'average rating': 8
                        #,'cuisine': 7
                        ,'vegetarians': 4
                        }
				   ,'The Fork':{'distance' : 3
                        ,'novelty' : 9
                        #,'cost': 2
                        #,'average rating': 7
                        ,'cuisine': 8
                        ,'vegetarians': 8
                        }
}

#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
rest_names = list(restaurants)
rest_cols =list(restaurants[rest_names[1]])

M_restaurants = np.zeros((len(rest_names),len(rest_cols)))
for i, r in enumerate(restaurants):
	M_restaurants[i,] = np.array(list(restaurants[r].values()))

print(M_restaurants)

#%%
# The most important idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is and how it will relate to our restaurant matrix.

"""
A linear combination is an expression constructed from a set of terms by multiplying each terms by a constant and summing the results. In this example our restaurant matrix is our set of terms.
For each person, we will multiply the corresponding category for each restaurant, by the weight assigned to that value. We will then summ those values to reach each persons score for a particular
restaurant, and determine our optimal choice.
"""

#%%
# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent?
p = people_names.index('Max')
out=np.dot(M_people[p,],M_restaurants.T)


print(out)

outRank = getRank(out)

print("Best Restaurant for", people_names[p],"is",rest_names[np.flatnonzero(outRank ==1)[0]],"with a score of",out[np.flatnonzero(outRank ==1)[0]])

"""
Each entry in this vector represents the overall score of each restaurant for Max, the person we chose.
"""
#%%
# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
M_usr_x_rest=np.dot(M_restaurants,M_people.T)

print(M_usr_x_rest)
print("Rows are restaurants: ",rest_names) ##
print("Columns are people: ",people_names) ##

"""
In this matrix, the rows are restaurants and the columns are people. Each value represents the score that a given person gave for a given restaurant.
"""
#%%
# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?
rest_score  = M_usr_x_rest.sum(axis=1) 
c=0
for i in reversed(np.argsort(rest_score)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", rest_score[i])  

rest_rank = getRank(rest_score)
rest_best_id = np.flatnonzero(rest_rank ==1)[0]
rest_best_score = rest_score[rest_best_id]
rest_best_name = rest_names[rest_best_id]

print("Best Restaurant for all people (based on Scores) is",rest_best_name,"with a score of", rest_best_score)

"""
Each entry in this vector represents the sum of each persons score for a given restaurant. 
"""
#%%
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  
M_usr_x_rest_rank = np.zeros_like(M_usr_x_rest)

for r in range(M_usr_x_rest.shape[1]):
    M_usr_x_rest_rank[:,r] = getRank(M_usr_x_rest[:,r])

rest_score2   = np.sum(M_usr_x_rest_rank,axis=1)
c=0
for i in (np.argsort(rest_score2)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", rest_score2[i]) #,"(", (120- rest_score2[i]),")")  

rest_rank2 = ss.rankdata(rest_score2,method='min')
rest_best_id2 = np.flatnonzero(rest_rank2 ==1)[0]
rest_best_score2 = rest_score2[rest_best_id2]
rest_best_name2 = rest_names[rest_best_id2]

print("Best Restaurant for all people (based on Ranks) is",rest_best_name2,"with a score of", rest_best_score2)
#%%
# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?

"""
There is a difference between the two because one is based on the raw scores obtained via weights of the survery results, and the other is based on an ordinal ranking system.
While the raw score takes into account the magnitude of each characteristic, it is vulnerable to outliers, which can cause skewness.
In contrast, the ordinal ranking system is not vulnerable to outliers, but does not take into account the magnitude of score difference.
In the real world, this represents the idea that not all features of a restaruant are equally important, or similarly important to different people.
"""
#%%
# How should you preprocess your data to remove this problem. 

"""
I would recommend a log transform for this particular data set because it would reduce skewness and the potential effects of any outliers.
"""
rest_score_log  = [np.log(M_usr_x_rest[x,:]).sum() for x in range(M_usr_x_rest.shape[0])]
c=0
for i in reversed(np.argsort(rest_score_log)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", round(rest_score_log[i],3))  

rest_rank_log = getRank(rest_score_log)
rest_best_id_log = np.flatnonzero(rest_rank_log ==1)[0]
rest_best_score_log = rest_score_log[rest_best_id_log]
rest_best_name_log = rest_names[rest_best_id_log]
 
print("Best Restaurant for all people (based on Logged Scores) is",rest_best_name_log,"with a score of", round(rest_best_score_log,3))
#%%
# Find  user profiles that are problematic, explain why?

"""
My ranks did not change based on the method, because The Fork ranked first on the majority of categories for all users. However, it could be problematic if there were more differences in the data.
"""
_=rest_names.index(rest_best_name)
print(rest_best_name ," scores",M_usr_x_rest[_,:]," Final Rank Method 1-Scores: ",rest_rank[_])
print(rest_best_name, " Ranks",M_usr_x_rest_rank[_,:]," Final Rank Method 2-Ranks: ",rest_rank2[_])
print('-'*10)
_=rest_names.index(rest_best_name2)
print(rest_best_name2, " scores",M_usr_x_rest[_,:]," Final Rank Method 1-Scores: ",rest_rank[_])
print(rest_best_name2, " ranks",M_usr_x_rest_rank[_,:]," Final Rank Method 2-Ranks: ",rest_rank2[_])

#%%
# Think of two metrics to compute the disatistifaction with the group.  

"""
My first metric takes the highest restaurant score for each person, and finds the difference between that and the score for a given restaurant.
The result is a ranked list, where we can determine who will be the most dissatisfied with a paricular choicem and the least dissatisfied.
This is useful for figuring out who will be happiest with a particular restaurant decision, and who will be the least happy.
"""
rest_best_max_score = M_usr_x_rest[rest_best_id,:].max()
people_dissatisfaction =  abs(M_usr_x_rest[rest_best_id,:]   - rest_best_max_score)
print("List of people from the most disattisfied to the least disattisfied about the choice of",rest_best_name,"as best restaurant")
i=0
for x in reversed(np.argsort(people_dissatisfaction) ) :
    i=i+1
    print(i,". ",people_names[x],"disattisfaction score:",people_dissatisfaction[x])
#------------
print("-"*20)
rest2_best_max_score = M_usr_x_rest[rest_best_id2,:].max()
people_dissatisfaction2 =  abs(M_usr_x_rest[rest_best_id2,:]   - rest2_best_max_score)
print("List of people from the most disattisfied to the least disattisfied about the choice of",rest_best_name2,"as best restaurant")
i=0
for x in reversed(np.argsort(people_dissatisfaction2) ) :
    i=i+1
    print(i,". ",people_names[x],"disattisfaction score:",people_dissatisfaction2[x])
    
print("We can measure the overall disattisfaction in the group by using a standard deviation of above values")
print(np.std(people_dissatisfaction2))
print("We have a large standard deviation, thus we can conclude there is a strong disatisfaction within the group on the selected restaurant")

"""
The second metric measures the difference between the average score for a particular restaurant vs. an individuals score for a particular restaurant.
The differences for each users score of a particular restaurant are averaged to achieve a single score for each restaurant.
Thus, we are able to see each restuarant ranked from most polarizing to least polarizing.
"""
rest_mean_score = M_usr_x_rest.mean(axis=1)

rest_people_disattisfaction  =  np.mean(abs(M_usr_x_rest.T - rest_mean_score),axis=0)
rest_people_disattisfaction_sd  =  np.std(abs(M_usr_x_rest.T - rest_mean_score),axis=0)

print("List of restaurants from the most disattisfaction to the least disattisfaction vs. peoples preference")
i=0
for x in reversed(np.argsort(rest_people_disattisfaction) ) :
    i=i+1
    print(i,".",rest_names[x],"overall disattisfaction:",round(rest_people_disattisfaction[x],2)," (Std:",round(rest_people_disattisfaction_sd[x],3),").")


print("We can see 10 Barrels and The Fork habe the highest disatisfaction score even if they were among the highest overall scores. This indicates\
      that there is disagreement within the team. Splitting the people in groups, and choose a best restaurant\
      per each group may help to reduce the overall disatisfaction")
#%%
# Should you split in two groups today? 
"""
I clustered the data into two groups to see if that would reduce dissatifaction, and produce different results. One group chose to go to The Fork, while another chose to go to Pizza Hut.
Both groups reduced their dissatisfaction score and standard deviation, leading me to believe the best option is to split the two groups.
"""
from sklearn.cluster import KMeans
X = M_usr_x_rest.T
k=2
km = KMeans(n_clusters=k, random_state=0,n_init=10,max_iter=300).fit(X)

def bestRest(data):
    rest_score_  = np.sum((data),axis=1)
    print(rest_score_)
       
    rest_rank_  = getRank(rest_score_)
    rest_best_id_  = np.flatnonzero(rest_rank_ ==1)[0]
    rest_best_score_  = rest_score_log[rest_best_id_]
    rest_best_name_  = rest_names[rest_best_id_]
     
    # Each entry represent the total value per each restaurant across all people 
    rest_mean_score_ = (data[rest_best_id_,:]).mean()
    
    diss  =  np.mean(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    diss_sd  =  np.std(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    #diss =  abs(data.mean(axis=1) - data.max(axis=1))[rest_best_id_]
    
    print("Best Restaurant for this group is ",rest_best_name_,"with a score of", round(rest_best_score_,3)
    ," disattisfaction score:",round(diss,2)," (std:",round(diss_sd,3),")")
    return(rest_best_id_)
    
print("-"*20)

for i in range(k):
    print("CLUSTER ",i)
    data=np.dot(M_restaurants,M_people[km.labels_ == i].T)   
    
    print("people:", data.shape[1]," ->",np.where(km.labels_==i))
    _=bestRest(data)
    print("      Original Disattisfaction Score for ",rest_names[_],":",rest_people_disattisfaction[_]," (std:",rest_people_disattisfaction_sd[_],").")
    print("-"*20)

print("We can see we have selected The Fork and Pizza Hut for the two groups. In both situations the disattisfaction score for the \
      selected restaurant decreased. The stardard deviations also decreased.")
#%%
# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?
"""
This changes Cost to the maximum value for all people because it is now covered by your boss. Thus I changed all Cost ratings to 5, the highest value.
However, even with this change, The Fork remains the top restaurant. This is because it ranked highly on cost to begin with, as well as ranking high in other areas.
"""
M_people2 = M_people.copy()
M_people2[:,people_cols.index('cost')]=5
M_usr_x_rest2=np.dot(M_people2,M_restaurants.T)

rest2_score  = [M_usr_x_rest2[:,x].sum() for x in range(M_usr_x_rest2.shape[1])]

for i in np.argsort(rest2_score):
    print("Restaurant",rest_names[i], " tot Score:", rest2_score[i])  

rest2_rank = getRank(rest2_score)
rest2_best_id = np.flatnonzero(rest2_rank ==1)[0]
rest2_best_score = rest2_score[rest2_best_id]
rest2_best_name = rest_names[rest2_best_id]

print("Best Restaurant for all people is",rest2_best_name,"with a score of", rest2_best_score)
#%%
# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
"""
We can use the final decision matrix to back into the weight matrix for the new team.
"""
#Assuming we have the following decision matrix for the restarants
new_team_x_rest =  M_usr_x_rest

new_team_matrix= np.linalg.lstsq(M_restaurants,new_team_x_rest,rcond=-1)[0].T

print("New Team Weight Matrix is:")
print(new_team_matrix)
for i,p in enumerate(people_names): print(p,"weights:",new_team_matrix[i,:])
print("Columns Name:")
print(people_cols)
"""
We can use the final score for each restaurant to calculate the final weight matrix for each person.
"""
new_team_rest_score = np.sum(new_team_x_rest,axis=1)
print("Final Score per each restarant")
for i,s in enumerate(new_team_rest_score): print(rest_names[i],"tot weight:",round(s,3))
new_team_tot_weights = np.linalg.lstsq(M_restaurants,new_team_rest_score.T,rcond=-1)[0]
print("\nSum of people weights per each criteria")
for i,s in enumerate(new_team_tot_weights): print(people_cols[i],"tot weight:",round(s,3))