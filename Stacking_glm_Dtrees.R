#Stacking implementation for classification of loan defaulters
#Models used are Decision trees(CART and C5.0) and logistic regression

rm(list=ls(all=TRUE))

library(RCurl)

data=read.table(text = getURL("https://raw.githubusercontent.com/rajsiddarth/Ensemble_Learning/master/dataset.csv"), header=T, sep=',',
                col.names = c('ID', 'age', 'exp', 'inc', 
                              'zip', 'family', 'ccavg', 'edu', 
                              'mortgage', 'loan', 'securities', 
                              'cd', 'online', 'cc'))
#Removing the id, zip and experience

data=subset(data,select = -c(ID,zip,exp))

#Numeric attributes : age,inc,family,CCAvg,Mortgage
#Categorical: Education,Securities account,CD Account,Online,Credit card
#Target Variable: Personal Loan

num_data=data.frame(sapply(data[c('age','inc','family','ccavg')],function(x){as.numeric(x)}))
categ_attributes=c('edu','securities','cd','online')
categ_data=data.frame(sapply(data[categ_attributes],function(x){as.factor(x)}))
ind_data=data.frame(sapply(data["loan"], as.factor))

#Convert numeric attributes to categorical attributes using equal frequency
#install.packages("infotheo")

library(infotheo)
convert_data=data.frame(sapply(num_data, function(x){infotheo::discretize(x,disc="equalfreq",nbins=4)}))
categ_data2=data.frame(sapply(convert_data,function(x){as.factor(x)}))
data=cbind(categ_data2,categ_data,ind_data)

#Dividing into train and test
library(caTools)
index=sample.split(data$loan,SplitRatio = 0.7)
train=data[index,]
test=data[!index,]
ind_variable=setdiff(names(data),"loan")

# Check how records are split with respect to target attribute.
table(data$loan)
table(train$loan)
table(test$loan)

#Stacking 

# Building CART model on the training dataset

library(rpart)
model_Cart = rpart(loan ~ ., train, method = "class")
summary(model_Cart)

# Building C5.0 model on the training dataset
library(C50)
model_c50 = C5.0(loan ~ ., train, rules = T)
summary(model_c50)

# Building Logistic regression on the training dataset
model_glm = glm(loan ~ ., train, family = binomial)
summary(model_glm)

#Predicting on train data

# Using CART Model predict on train data
train_cart = predict(model_Cart, train, type = "vector") 
table(train_cart)

# if we choose type=vector, then replace 1 with 0 and 2 with 1
train_cart = ifelse(train_cart == 1, 0, 1)
table(train_cart)

# Using C5.0 Model to predict on train data set
train_c50 = predict(model_c50, train, type = "class")
train_c50 = as.vector(train_c50)
table(train_c50)

# Using GLM Model to predict on train dataset
train_glm = predict(model_glm, train, type = "response")


#it gives probabilities, so we #need to convert to 1's and 0's; 
# if >0.5 show as 1 or else show as 0.
train_glm = ifelse(train_glm > 0.5, 1, 0) 
table(train_glm)

# Combining training predictions of CART, C5.0 & Logistic Regression together
Combine_train_model = data.frame(CART = train_cart,C50 = train_c50,GLM = train_glm)
Combine_train_model = data.frame(sapply(Combine_train_model, as.factor))
str(Combine_train_model)


# Viewing the predictions of each model
table(Combine_train_model$CART) #CART 
table(Combine_train_model$C50)  #C5.0
table(Combine_train_model$GLM)  #Logistic Regression
table(train$loan) #Original Dataset DV

# Adding the actual target to the dataframe
Combine_train_model= cbind(Combine_train_model, loan = train$loan)

# Ensemble Model with GLM as combiner

model_ensemble = glm(loan ~ ., Combine_train_model, family = binomial)
summary(model_ensemble)

# Check the "ensemble_Model model" on the train data
train_ensemble = predict(model_ensemble, Combine_train_model,type = "response")
train_ensemble = ifelse(train_ensemble > 0.5, 1, 0)
table(train_ensemble)

cmatrix_ensemble = table(train_ensemble, Combine_train_model$loan)

train_accuracy=sum(diag(cmatrix_ensemble))/sum(cmatrix_ensemble)
cat("accuracy on train data= ",round(train_accuracy,3)*100)

#Test data 
# Using CART Model to predict on test dataset
test_Cart = predict(model_Cart, test, type="vector")
test_Cart = ifelse(test_Cart == 1, 0, 1)

# Using C50 Model to predict on test dataset
test_c50 = predict(model_c50, test, type = "class")
test_c50 = as.vector(test_c50)

# Using GLM Model prediction on test dataset
test_glm = predict(model_glm, test, type="response")
test_glm = ifelse(test_glm > 0.5, 1, 0)


# Combining test predictions of CART, C5.0 & Logisting Regression 
combine_test_model = data.frame(CART = test_Cart, C50 = test_c50, GLM = test_glm) 
combine_test_model = data.frame(sapply(combine_test_model, as.factor))
str(combine_test_model)

# "glm_ensemble model" on the test data
test_ensemble = predict(model_ensemble, combine_test_model, type = "response")
test_ensemble = ifelse(test_ensemble > 0.5, 1, 0)
table(test_ensemble)

#Calculating accuracy on test data using ensemble model
cmatrix_ensemble = table(test_ensemble, test$loan)

test_accuracy=sum(diag(cmatrix_ensemble))/sum(cmatrix_ensemble)
cat("accuracy on test data= ",round(test_accuracy,3)*100)


