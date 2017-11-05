import pandas as pd
import numpy  as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import re

###########################################################################################################################3
#In this section
#list of stop words which can used to a reference list for data cleansing in the user skill column
#list of educational qualification & certification (list may not be very exhaustive)

stopword_list  = stopwords.words("english")
list_remove_user_skills = ["yr","yrs" ,"yr","&amp",";","year","[" ,"]" ,"yes","month","months","!","?" ,"experience" ,"other",":"
                            "english","hindi","sanskrit","punjabi","typing","datacard","mts","sales","customer","business development"
                            "channel development","price","price management","customer relation"]
                            
educational_list = ["bca","mca","b.tech","m.tech","bsc","msc","computer science","mba-it","engineering","phd"]
cert_list        = ['ccnp', 'ccie', 'jncie-ent', 'comptia network+', 'wcna', 'dcnt', 'mcsa', 'ccna', 'rhca ', 
                      'mcsa', 'oca', 'ocp', 'ocpjp', 'aix', 'a+', 'n+', 'jncia', 'rhce', 'vca/vcp', 'vca', 'vcp',"mcp"]
#################################################################################################################################3
#In this section
#Update stopword_list with list_remove_user_skills , educational_list & cert_list list 
for i in list_remove_user_skills:
    stopword_list.append(i)
for i in educational_list:
    stopword_list.append(i)
for i in cert_list:
    stopword_list.append(i)
##################################################################################################################################
#read the source file
df = pd.read_excel(r"C:\Users\sauravghosh\Desktop\Machine_Learning\HT Media\Data\Problem_Data.xlsx",sheetname = "Train_Data")
#create placeholder for new features with default values
df["Relevant_Degree"] = 0
df["Releveant_Cert"]  = 0
df["Industry"]        = np.NaN
df["Sector"]          = np.NaN
#get the column names
col_names = df.columns
##########################################################################################################################
#In this section
#The func user_skill_len_flg was used to convert all the values in user skill columns whose length was less than 3 
#to null because most of such values contained no information related to skills.
def user_skill_len_flg(x):
    skills_to_include = ["c","c++","sap","php","aix","cad"]
    if pd.notnull(x) :
        user_skill_flg = len(str(x).strip())
        if x not in skills_to_include :
            if user_skill_flg <= 3 :
                return 1
            else:
                return 0

df["User_Len_Flg"] = df.apply(lambda x :user_skill_len_flg(x["User_Skills"]),axis = 1)
df["User_Skills"][df["User_Len_Flg"] == 1] = np.NaN
df.drop("User_Len_Flg",inplace = True , axis = 1)
################################################################################################################################3
# In this section
#remove spaces from the user skill column
dataset = np.array(df)
for d in range(len(dataset)):
    if pd.notnull(dataset[d][6]) and np.isreal(dataset[d][6]) != True :
        word = dataset[d][6].split(",")
        word = [i.strip() for i in word]
        dataset[d][6] = ",".join([w.strip() for w in word])

df = pd.DataFrame(dataset)
df.columns  = col_names
del dataset
################################################################################################################################
#In this section 
#1. Convert all the rows in user skill and user profile title column to null if it only has numbers
#2. Merge the text information from user skill and user job title columns
#3. Extract the relevant degree information by comparing the merged text with the user profile title column based on fuzzy logic match of greater than 70%
#4. Extract the relevant certification information by comparing the merged text with the user profile title column based on fuzzy logic match of greater than 70%  
#5. Extract  year of exp from User_JobTitle column for corresponding missing values in user exp column using regex . I realized later we could have used the merged text , 
#   this can be area of improvement and further tunning is required in the regex expression for better results.
#6. Extract industry and sector information from the User_Industry column. The text was splitted and the first value was assumed
#   to be industry and the rest of the values where merged to get the sector information


dataset = np.array(df)
for d in range(len(dataset)):
    #if the cell value consist of only numbers then replace it with null  
    if pd.notnull(dataset[d][6]) :
        user_skill_text = str(dataset[d][6])  
        user_profile_text = str(dataset[d][3])    
        user_skill_reg  = re.findall(r"[A-Za-z]+",user_skill_text)
        user_profile_reg = re.findall(r"[A-Za-z]+",user_profile_text)       
        if len(user_skill_reg) == 0 :
            dataset[d][6] = np.NaN       
        if len(user_profile_reg) == 0:
            dataset[d][3] = np.NaN
    
    #merge the text information from user skill and user job title columns
    if pd.notnull(dataset[d][6]) :
        user_skills = dataset[d][6].split(",")
        if pd.notnull(dataset[d][3]):
            user_profile = dataset[d][3].split(",")
            user_skills = np.append(user_skills,user_profile)
            
    #get the relevant degree information    
        if len(user_skills) > 0 :
            for w in user_skills:
                for e in  educational_list:
                    match_score = fuzz.token_set_ratio(e,w.strip())
                    if match_score >= 80 :
                        dataset[d][7] = 1
                        break
    #get the relevant certification information   
                for c in cert_list:
                    match_score = fuzz.token_set_ratio(c,w.strip())
                    if match_score >= 80 :
                        dataset[d][8] = 1
                        break
            
    #get year of exp from User_JobTitle column for corresponding missing values in user exp column
            if pd.isnull(dataset[d][5]) :
                if pd.notnull(dataset[d][3]) :
                    text = re.findall(r"[0-9][\s+year|year]",dataset[d][3])
                    if len(text) > 0 :
                        text = " ".join([t for t in text])
                        text = re.findall(r"[0-9]",text)
                        text = max(text)
                        dataset[d][5] = text
            
     #extract industry and sector information
            if pd.notnull(dataset[d][4]):
                text = dataset[d][4].split()
                dataset[d][9] = text[0] #get industry
                if len(text[1:]) > 0 :
                    dataset[d][10] = "".join([i.strip() for i in text[1:]]) #get sector
                else:
                    dataset[d][10] = text[0]
                
df = pd.DataFrame(dataset)
df.columns = col_names
del dataset 

##############################################################################################################################3
#In this section
#1 . Impute missing values function is used to impute the missing values for user exp ., industry and sector columns
#2.  The approach taken is a proxy, where for each sub function area , I calculated median , mode for user exp ., industry and sector columns respectively.
#    And thereafter imputed the missing values for each such function area

def impute_missing_values(df,industry_list):
    for i in industry_list :

        median_wrk_exp = df["User_Experience (Years)"][df["User_SubFunctional_Area"] == i].median()        
        industry_mode  = df["Industry"][df["User_SubFunctional_Area"] == i].value_counts().index[0]
        sector_mode    = df["Sector"][(df["Industry"] == industry_mode) & (df["User_SubFunctional_Area"] == i)].value_counts().index[0]
    
        df["Industry"][(df["User_SubFunctional_Area"] == i) & (df["Industry"].isnull())] = industry_mode
        df["Sector"][(df["User_SubFunctional_Area"] == i) & (df["Sector"].isnull())] = sector_mode 
        df["User_Experience (Years)"][(df["User_SubFunctional_Area"] == i) & (df["User_Experience (Years)"].isnull())] = median_wrk_exp

    return df
    
df["User_Experience (Years)"][df["User_Experience (Years)"] == "<.01"] = 0
industry_list  = df.User_SubFunctional_Area.value_counts().index
df = impute_missing_values(df,industry_list)
##################################################################################################################################
#In this section
#1.I have used python's sklearn libary TfidfVectorizer to get word list with weighted scores from the user skill column
#2.I have used fuzzy logic algo with a threshold of 70% to standardize the words in the extracted list.
#3.I have updated the standard list words back to user skill columns based on exact match


Tfidf_Vectorizer = TfidfVectorizer(max_df = .95 , min_df = 2 , stop_words = stopword_list , norm = "l2",lowercase = True)
Tfidf_Vectorizer.fit_transform(df["User_Skills"][(df["User_Skills"].notnull())])
tfidf_feature_names = Tfidf_Vectorizer.get_feature_names()

#store the standard values word_list values
word_list = {}
for t in tfidf_feature_names:
    for i in tfidf_feature_names:
        if t == i :
            continue
        match_score = fuzz.token_set_ratio(i,t)
        if match_score > 70 :
            if len(t) < len(i):
                word_list[str(t.strip())] = str(i.strip())

#update the user skill column with stardard values
dataset = np.array(df)
search_keys = [str(w) for w in word_list.keys()]
for d in range(len(dataset)) :
    if pd.notnull(dataset[d][6]) :
        words = dataset[d][6].split(",")
        for w in words :
            if w.strip() in word_list.keys():
                dataset[d][6] = dataset[d][6].replace(w,word_list[w.strip()])

######################################################################################################################################3

df = pd.DataFrame(dataset)
df.columns = col_names
del dataset
df.to_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\HT Media\Data\Topic_Modelling_Dataset\topic_modelling.csv",index = False)




