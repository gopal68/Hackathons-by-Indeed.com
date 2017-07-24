#Title: "Indeed Machine Learning CodeSprint" 
#Challenge: "Tagging Raw Job Descriptions"
#Source: https://www.hackerrank.com/indeed-ml-codesprint-2017 
#Author: Gopal Gupta
#Date: 7 April 2017'

#Input : total 2 files ()
#file1: train.tsv , number of columns =2(tags,desc) , number of rows = 4375 ( exclude header)
#file2: test.tsv , number of columns =1(desc) , number of rows = 2921 ( exclude header)


#output : 1 file.
#file1: tags.tsv , number of columns =1(tags) , number of rows = 2921 ( exclude header)


#*****Note: this is not an optimize Script *******


#install packages 
install.packages("tm")
install.packages("xgboost")
install.packages("caret")
install.packages("dplyr")



# Load packages
library("tm")
library("xgboost")  # the main algorithm
library("caret")    # for the confusionmatrix() function (also needs e1071 package)
library("dplyr")    # for some data preperation


#  Keep your dataset in the current working directory or set the working directory where datasets are present. 

getwd()
#setwd()

# Read the train.tsv file - provided by Hacker rank (https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions)

train <- readLines("train.tsv")

#check the structure of your train dataset
str(train)

#total lines in dataset 4376 including Header
# every line having tags separated by space and job description split by '\t'

#check summary of your data
summary(train)

# preprocess the data - 


# We have total 12 types of tag, I am giving new name to tags, tag1 to tag12 as mentioned below
#tag1  = part-time-job
#tag2  = full-time-job
#tag3  = hourly-wage
#tag4  = salary
#tag5  = associate-needed
#tag6  = bs-degree-needed
#tag7  = ms-or-phd-needed
#tag8  = licence-needed
#tag9  = 1-year-experience-needed
#tag10 = 2-4-years-experience-needed
#tag11 = 5-plus-years-experience-needed
#tag12 = supervising-job



# creating data frame with 13 values - 1 description +12 tags 
# if tag is exist for description than will set 1 in respectve tag columns

# creating data frame with 13 variables, 1 observation to create data frame
train_df <- data.frame(desc="",tag1=0,tag2=0,tag3=0,tag4=0,tag5=0,tag6=0,tag7=0,tag8=0,tag9=0,tag10=0,tag11=0,tag12=0, stringsAsFactors = F)

# now we have data frame with 1 observation and 13 variable variables 

# delete 1 observation
train_df <- train_df[-1,]
str(train_df)

#intialize all tags with 0 and description with NA for all records 
train_df[1:4375,2:13] <- 0 
train_df[1:4375,1] <- NA
str(train_df)
summary(train_df)  

#now perform below function to fill data frame when tag exist in respective tag columns   

# first splitting the tags and descriptions by '\t'
# second splitting tags by space 
for(i in 1:4375) { 
  fields = strsplit(train[i+1], '\t')[[1]]
  
  train_df[i,1] = fields[2]
  # fields[1]
  if(fields[1] != ""){
    tags = strsplit(fields[1],' ')[[1]]
    
    
    l<-length(tags)
    
    # print(tags)
    for(j in 1:l)
    { 
      x<-switch(tags[j],
                "part-time-job"                  = "tag1",
                "full-time-job"                  = "tag2",
                "hourly-wage"                    = "tag3",
                "salary"                         = "tag4",
                "associate-needed"               = "tag5",
                "bs-degree-needed"               = "tag6",
                "ms-or-phd-needed"               = "tag7",
                "licence-needed"                 = "tag8",
                "1-year-experience-needed"       = "tag9",
                "2-4-years-experience-needed"    = "tag10",
                "5-plus-years-experience-needed" = "tag11",
                "supervising-job"="tag12"
      )
      train_df[i,x] <- 1
    }
  }
  
}

# check data frame structure
print(train_df[1:10,2:13])

# validate manually for few records here I am validating for 10th record 
train[11]

# find mutually exclusive class by plot,table
#plot(train_df[,2:13])

# I found max 5 tags are possible.
#one from tag1 or tag2 
#one from tag3 - tag4
#one from tag5 - tag6 - tag7 - tag8 
#one from tag9 - tag10 - tag11 
#one from tag12


#I am making 5 vectors for every type of tags 

#tag_1_2 - total 3 class 0,1,2
tag_1_2 <- 0
tag_1_2[1:4375] <- 0
tag_1_2[train_df$tag1 ==1] <- 1
tag_1_2[train_df$tag2 ==1] <- 2

#tag_3_4 -total 3 class 0,1,2

tag_3_4 <- 0
tag_3_4[1:4375] <- 0
tag_3_4[train_df$tag3 ==1] <- 1
tag_3_4[train_df$tag4 ==1] <- 2

#tag_5_6_7_8 - total 5 class 0,1,2,3,4

tag_5_6_7_8  <- 0
tag_5_6_7_8[1:4375] <- 0
tag_5_6_7_8[train_df$tag5 ==1] <- 1
tag_5_6_7_8[train_df$tag6 ==1] <- 2
tag_5_6_7_8[train_df$tag7 ==1] <- 3
tag_5_6_7_8[train_df$tag8 ==1] <- 4


#tag_9_10_11 - total 4 class 0,1,2,3
tag_9_10_11  <- 0
tag_9_10_11[1:4375] <- 0
tag_9_10_11[train_df$tag9  ==1] <- 1
tag_9_10_11[train_df$tag10 ==1] <- 2
tag_9_10_11[train_df$tag11 ==1] <- 3

#tag_12  total 2 class 0,1 ( binary classification ) 
tag_12  <- train_df$tag12




# read the test tsv file and store all test data set description in one data frame
test<- readLines("test.tsv")


# create data frame for test description 
test_df <- data.frame(desc=test[2:2922],stringsAsFactors = F)

# merge the train & test data set, total rows now = 7296

train_plus_test_desc <- train_df$desc[1:4375]
train_plus_test_desc <- c(train_plus_test_desc,test_df$desc) 



# apply text mining methods to create Sparse matrix 

# create Corpus data from our train & test data 
corpus_m <- Corpus(VectorSource(train_plus_test_desc))
# converting all the descriptions content in lower character  
corpus_m <- tm_map(corpus_m,tolower)

# removing Punctuation from the descriptions
corpus_m <- tm_map(corpus_m,removePunctuation)

# removing Words from the descriptions
corpus_m <- tm_map(corpus_m,removeWords,stopwords("english"))

# Stem words in the descriptions
corpus_m <- tm_map(corpus_m,stemDocument)

# Creating frequency matrix for corpus data
freq <- DocumentTermMatrix(corpus_m)

# find the list of words with lower frequency 
#findFreqTerms(freq, lowfreq = 50)

#Remove Sparse Terms from a freq DocumentTermMatrix
full_DTM <- removeSparseTerms(freq,0.99)
full_DTM

# converting DocumentTermMatrix to matrix  
full_matrix <- as.matrix(full_DTM)

# Split the data back into a train set and a test set matrixs.
train_matrix  <- full_matrix[1:4375,]
test_matrix <- full_matrix[4376:7296,]


####################################################################
# now we are ready to create models                          
# We will create below 5 models to predict the tags  
#
#1> model_1_2 to predict tag from tag1/tag2 - XGBOOST
#2> model_3_4 to predict tag from tag3/tag4 - XGBOOST
#3> model_5_6_7_8 to predict tag from tag5/tag6/tag7/tag8 - XGBOOST
#4> model_9_10_11 to predict tag from tag9/tag10/tag11 - XGBOOST
#5> model_12 to predict tag12 - XGBOOST
###############
#####################################################



#***************************************************************
#  Model name : cv_model_1_2
#  Input name : train_matrix + tag_1_2
#  output matrix : 
#***************************************************************

#create xgb  matrix for model_1_2
train_matrix_1_2 <- xgb.DMatrix(data = train_matrix, label = tag_1_2)

# find the unique classes to predict 
numberOfClasses <- length(unique(tag_1_2))

# set the xgb parameters 
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# set the number of XGBoost rounds
nround    <- 3 


# create the model 
model_1_2 <- xgb.train(params = xgb_params,
                       data = train_matrix_1_2, #change data set for every matrix 
                       nrounds = nround)

#check the model accuracy 
train_pred_1_2 <- predict(model_1_2, newdata = train_matrix_1_2)
train_prediction_1_2 <- matrix(train_pred_1_2, ncol = numberOfClasses,
                               nrow = length(train_pred_1_2)/numberOfClasses , byrow = T) 
head(train_prediction_1_2)
train_tag_1_2<-max.col(train_prediction_1_2 ) - 1

confusionMatrix(factor(tag_1_2),factor(train_tag_1_2),mode = "everything") 


#prediction  for test data set 
#create xgb  matrix for model_1_2
test_matrix_1_2 <- xgb.DMatrix(data = test_matrix)

test_pred_1_2 <- predict(model_1_2, newdata = test_matrix_1_2)
test_prediction_1_2 <- matrix(test_pred_1_2, ncol = numberOfClasses,
                              nrow = length(test_pred_1_2)/numberOfClasses , byrow = T) 
test_tag_1_2<-max.col(test_prediction_1_2) -1


#create matrix with actual tag
vec_tag_1_2 <- test_tag_1_2

vec_tag_1_2[test_tag_1_2==0] = ""
vec_tag_1_2[test_tag_1_2==1] = "part-time-job"
vec_tag_1_2[test_tag_1_2==2] = "full-time-job"

#***************************************************************
#   END --> model_1_2      
#***************************************************************






#***************************************************************
#  Model name : cv_model_3_4
#  Input name : train_matrix + tag_3_4
#  output matrix : 
#***************************************************************

#create xgb  matrix for model_3_4
train_matrix_3_4 <- xgb.DMatrix(data = train_matrix, label = tag_3_4)

# find the unique classes to predict 
numberOfClasses <- length(unique(tag_3_4))

# set the xgb parameters 
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# set the number of XGBoost rounds
nround    <- 3 


# create the model 
model_3_4 <- xgb.train(params = xgb_params,
                       data = train_matrix_3_4, #change data set for every matrix 
                       nrounds = nround)

#check the model accuracy 
train_pred_3_4 <- predict(model_3_4, newdata = train_matrix_3_4)
train_prediction_3_4 <- matrix(train_pred_3_4, ncol = numberOfClasses,
                               nrow = length(train_pred_3_4)/numberOfClasses , byrow = T) 
head(train_prediction_3_4)
train_tag_3_4<-max.col(train_prediction_3_4 ) - 1

confusionMatrix(factor(tag_3_4),factor(train_tag_3_4),mode = "everything") 


#prediction  for test data set 
#create xgb  matrix for model_3_4
test_matrix_3_4 <- xgb.DMatrix(data = test_matrix)

test_pred_3_4 <- predict(model_3_4, newdata = test_matrix_3_4)
test_prediction_3_4 <- matrix(test_pred_3_4, ncol = numberOfClasses,
                              nrow = length(test_pred_3_4)/numberOfClasses , byrow = T) 
test_tag_3_4<-max.col(test_prediction_3_4) -1


#create matrix with actual tag
vec_tag_3_4 <- test_tag_3_4

vec_tag_3_4[test_tag_3_4==0] = ""
vec_tag_3_4[test_tag_3_4==1] = "hourly-wage"
vec_tag_3_4[test_tag_3_4==2] = "salary"


#***************************************************************
#   END --> model_3_4      
#***************************************************************





#***************************************************************
#  Model name : cv_model_5_6_7_8
#  Input name : train_matrix + tag_5_6_7_8
#  output matrix : 
#***************************************************************

#create xgb  matrix for model_5_6_7_8
train_matrix_5_6_7_8 <- xgb.DMatrix(data = train_matrix, label = tag_5_6_7_8)

# find the unique classes to predict 
numberOfClasses <- length(unique(tag_5_6_7_8))

# set the xgb parameters 
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# set the number of XGBoost rounds
nround    <- 3 


# create the model 
model_5_6_7_8 <- xgb.train(params = xgb_params,
                           data = train_matrix_5_6_7_8, #change data set for every matrix 
                           nrounds = nround)

#check the model accuracy 
train_pred_5_6_7_8 <- predict(model_5_6_7_8, newdata = train_matrix_5_6_7_8)
train_prediction_5_6_7_8 <- matrix(train_pred_5_6_7_8, ncol = numberOfClasses,
                                   nrow = length(train_pred_5_6_7_8)/numberOfClasses , byrow = T) 
head(train_prediction_5_6_7_8)
train_tag_5_6_7_8<-max.col(train_prediction_5_6_7_8 ) - 1

confusionMatrix(factor(tag_5_6_7_8),factor(train_tag_5_6_7_8),mode = "everything") 


#prediction  for test data set 
#create xgb  matrix for model_5_6_7_8
test_matrix_5_6_7_8 <- xgb.DMatrix(data = test_matrix)

test_pred_5_6_7_8 <- predict(model_5_6_7_8, newdata = test_matrix_5_6_7_8)
test_prediction_5_6_7_8 <- matrix(test_pred_5_6_7_8, ncol = numberOfClasses,
                                  nrow = length(test_pred_5_6_7_8)/numberOfClasses , byrow = T) 
test_tag_5_6_7_8<-max.col(test_prediction_5_6_7_8) -1


#create matrix with actual tag
vec_tag_5_6_7_8 <- test_tag_5_6_7_8

vec_tag_5_6_7_8[test_tag_5_6_7_8==0] = ""
vec_tag_5_6_7_8[test_tag_5_6_7_8==1] = "associate-needed" 
vec_tag_5_6_7_8[test_tag_5_6_7_8==2] = "bs-degree-needed" 
vec_tag_5_6_7_8[test_tag_5_6_7_8==3] = "ms-or-phd-needed"
vec_tag_5_6_7_8[test_tag_5_6_7_8==4] = "licence-needed" 

#***************************************************************
#   END --> model_5_6_7_8      
#***************************************************************




#***************************************************************
#  Model name : cv_model_9_10_11
#  Input name : train_matrix + tag_9_10_11
#  output matrix : 
#***************************************************************

#create xgb  matrix for model_9_10_11
train_matrix_9_10_11 <- xgb.DMatrix(data = train_matrix, label = tag_9_10_11)

# find the unique classes to predict 
numberOfClasses <- length(unique(tag_9_10_11))

# set the xgb parameters 
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# set the number of XGBoost rounds
nround    <- 3 


# create the model 
model_9_10_11 <- xgb.train(params = xgb_params,
                           data = train_matrix_9_10_11, #change data set for every matrix 
                           nrounds = nround)

#check the model accuracy 
train_pred_9_10_11 <- predict(model_9_10_11, newdata = train_matrix_9_10_11)
train_prediction_9_10_11 <- matrix(train_pred_9_10_11, ncol = numberOfClasses,
                                   nrow = length(train_pred_9_10_11)/numberOfClasses , byrow = T) 
head(train_prediction_9_10_11)
train_tag_9_10_11<-max.col(train_prediction_9_10_11 ) - 1

confusionMatrix(factor(tag_9_10_11),factor(train_tag_9_10_11),mode = "everything") 


#prediction  for test data set 
#create xgb  matrix for model_9_10_11
test_matrix_9_10_11 <- xgb.DMatrix(data = test_matrix)

test_pred_9_10_11 <- predict(model_9_10_11, newdata = test_matrix_9_10_11)
test_prediction_9_10_11 <- matrix(test_pred_9_10_11, ncol = numberOfClasses,
                                  nrow = length(test_pred_9_10_11)/numberOfClasses , byrow = T) 
test_tag_9_10_11<-max.col(test_prediction_9_10_11) -1


#create matrix with actual tag
vec_tag_9_10_11 <- test_tag_9_10_11

vec_tag_9_10_11[test_tag_9_10_11==0] = ""
vec_tag_9_10_11[test_tag_9_10_11==1] = "1-year-experience-needed" 
vec_tag_9_10_11[test_tag_9_10_11==2] = "2-4-years-experience-needed" 
vec_tag_9_10_11[test_tag_9_10_11==3] = "5-plus-years-experience-needed"


#***************************************************************
#   END --> model_9_10_11      
#***************************************************************



#***************************************************************
#  Model name : cv_model_12
#  Input name : train_matrix + tag_12
#  output matrix : 
#***************************************************************

#create xgb  matrix for model_12
train_matrix_12 <- xgb.DMatrix(data = train_matrix, label = tag_12)

# find the unique classes to predict 
numberOfClasses <- length(unique(tag_12))

# set the xgb parameters 
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# set the number of XGBoost rounds
nround    <- 3 


# create the model 
model_12 <- xgb.train(params = xgb_params,
                      data = train_matrix_12, #change data set for every matrix 
                      nrounds = nround)

#check the model accuracy 
train_pred_12 <- predict(model_12, newdata = train_matrix_12)
train_prediction_12 <- matrix(train_pred_12, ncol = numberOfClasses,
                              nrow = length(train_pred_12)/numberOfClasses , byrow = T) 
head(train_prediction_12)
train_tag_12<-max.col(train_prediction_12 ) - 1

confusionMatrix(factor(tag_12),factor(train_tag_12),mode = "everything") 


#prediction  for test data set 
#create xgb  matrix for model_12

test_matrix_12 <- xgb.DMatrix(data = test_matrix)

test_pred_12 <- predict(model_12, newdata = test_matrix_12)
test_prediction_12 <- matrix(test_pred_12, ncol = numberOfClasses,
                             nrow = length(test_pred_12)/numberOfClasses , byrow = T) 
test_tag_12<-max.col(test_prediction_12) -1



#create matrix with actual tag
vec_tag_12 <- test_tag_12

vec_tag_12[test_tag_12==0] = ""
vec_tag_12[test_tag_12==1] = "supervising-job"


#***************************************************************
#   END --> model_12      
#***************************************************************

####merge all the tags and delete extra spaces 
tags_1_to_4= trimws(paste(vec_tag_1_2,vec_tag_3_4))
tags_1_to_8 = trimws(paste(tags_1_to_4,vec_tag_5_6_7_8))
tags_1_to_11 = trimws(paste(tags_1_to_8,vec_tag_9_10_11))
tags_1_to_12 = trimws(paste(tags_1_to_11,vec_tag_12))


final_output <- data.frame("tags"=tags_1_to_12)

write.table(final_output ,file="tags.tsv", quote=FALSE,row.names = F)

rm(list = ls())


###############################################################
# END
###############################################################


