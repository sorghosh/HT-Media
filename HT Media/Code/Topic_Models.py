# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 16:24:41 2017

@author: sauravghosh
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
import numpy as np

def display_topics(H, W, feature_names,no_top_words):
    topic_result = []
    topic_text = ""
    for topic_idx, topic in enumerate(H):
        print "Topic %d:" % (topic_idx)
        for  i in topic.argsort()[:-no_top_words -1:-1]:
            topic_text =  topic_text + " " +  " ".join([feature_names[i]])
        print topic_text
        topic_result.append(topic_text)
        topic_text = ""

    return topic_result

######################################################################################################################
#In this section
#reading the dataset for topic modelling
#creating placeholder of relevant skills and relevant match score columns
df = pd.read_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\HT Media\Data\Topic_Modelling_Dataset\topic_modelling.csv")
industry_list         = df.User_SubFunctional_Area.value_counts().index
df["Relevant_Skills"] = 0
df["Relevant_Match_Score"]  = 0
columns               = df.columns
dataset               = np.array(df)
###############################################################################################################################3
#In this section
#creating stopword list which will be used during topic modelling
stopword_list           = stopwords.words("english")
list_remove_user_skills = ["yr","yrs" ,"yr","&amp",";","year","[" ,"]" ,"yes","month","months","!","?" ,"experience" ,"other",":"
                            "english","hindi","sanskrit","punjabi","typing","datacard","mts","sales","customer","business"
                            "channel","price","price management","customer","accounts","office","tally"]
                            
educational_list        = ["bca","mca","b.tech","m.tech","bsc","msc","computer science","mba-it","engineering","phd"]
cert_list               = ['ccnp', 'ccie', 'jncie-ent', 'comptia network+', 'wcna', 'dcnt', 'mcsa', 'ccna', 'rhca ', 
                             'mcsa', 'oca', 'ocp', 'ocpjp', 'aix', 'a+', 'n+', 'jncia', 'rhce', 'vca/vcp', 'vca', 'vcp',"mcp"]

#update stopword_list 
for i in list_remove_user_skills:
    stopword_list.append(i)
for i in educational_list:
    stopword_list.append(i)
for i in cert_list:
    stopword_list.append(i)
###############################################################################################################################
#In this section 
####remove the stopwords from user skills columns based on fuzzy logic , this could be have been done in data cleaning code.
for d in range(len(dataset)):
    if pd.notnull(dataset[d][6]):    
        words = dataset[d][6].split(",")    
        for w in words :
            for s in stopword_list:
                match_score = fuzz.token_set_ratio(w, s)
                if match_score > 70 :
                    dataset[d][6] = dataset[d][6].replace(w,"")
##############################################################################################################################
#In this section

#1.Topic modelling is performed for each sub function.The objective was to find out key topics for each sub function and match it with
#  user skill column based on fuzzy logic at threshold of 70% (this can vary) to find out whether the mentioned skill is relevant and by what percent.
#2.Intutively 25 topics was build for each sub function and top 40 words
                  
                    
for i in industry_list :
    print i
    Tfidf_Vectorizer    = TfidfVectorizer(max_df = .95 , min_df = 2 , stop_words = stopword_list , norm = "l2",lowercase = True)
    tfidf               =  Tfidf_Vectorizer.fit_transform(df["User_Skills"][(df["User_SubFunctional_Area"] == i) & (df["User_Skills"].notnull())])
    tfidf_feature_names = Tfidf_Vectorizer.get_feature_names()
    
    no_topics = 25
    nmf_model = NMF(n_components=no_topics, random_state=1, init='nndsvd').fit(tfidf)
    nmf_W     = nmf_model.transform(tfidf)
    nmf_H     = nmf_model.components_
    no_top_words = 40
    skill_set = display_topics(nmf_H, nmf_W, tfidf_feature_names,no_top_words)
    for d in range(len(dataset)):
        if pd.notnull(dataset[d][6]):
#            score_list = [fuzz.token_set_ratio(i,dataset[d][6]) for i in skill_set]
#            match_score = sum(score_list)/float(len(score_list))
            match_score = fuzz.token_set_ratio(i,dataset[d][6])
            if match_score > 70 :
                dataset[d][11] = 1
                dataset[d][12] = match_score/float(100)

df = pd.DataFrame(dataset)
df.columns = columns 
del dataset
#######################################################################################################################
#This section should have been part of data cleansing. In this section I have converted the experience column into buckets
#based on the logic of 0-6 yrs as below assistant manager exp , 6- 12yrs mid management exp and above that top managment exp. Although
#I feel further feature extraction could be done for this feature.
#converting work experience column into category
def exp_bucket(x):
    if x >12 :
        return "12+"
    elif x >= 6 & x <= 12 :
        return "6-12 exp"
    elif x < 6 :
        return "less than 6yrs exp"

df["User_Experience (Years)"] = df["User_Experience (Years)"].astype(int)
df["User_Experience (Years)"] = df.apply(lambda x : exp_bucket(x["User_Experience (Years)"]),axis = 1)
########################################################################################################################################3
#save the processed dataset will all columns
df.to_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\HT Media\Data\Modelling_Dataset\Modelling_Dataset_All_Columns_V1.csv")
###################################################################################################################################333
#preparing the dataset for machine learning
# I have only used dataset where the revelant skills is equal to one for modelling work,the entire dataset is stored in Modelling_Dataset_All_Columns_V1.csv
# file. For machine learning I have used Modelling_DatasetV1.csv
df = df[df["Relevant_Skills"] != 0]
modelling_cols = ["User_SubFunctional_Area","User_Experience (Years)","Relevant_Degree" ,"Releveant_Cert","Industry","Sector","Relevant_Match_Score"]
df             = df[modelling_cols]
df.to_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\Times_Job\Data\Modelling_Dataset\Modelling_DatasetV1.csv",index = False)