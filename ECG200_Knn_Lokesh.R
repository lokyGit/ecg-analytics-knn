library(knnGarden)
library(caret)
library(e1071)

#loading the test data set
ECG_TEST = readxl::read_excel("/Users/lokeshpalacharla/Library/Mobile Documents/com~apple~CloudDocs/NEU/Classes/Summer 2020/Predictive Analytics/Week 1/ECG200/ECG200_TEST.xlsx", col_names = FALSE)

#getting the dimensions of the test set
dim(ECG_TEST)

#Viewing the test set
View(ECG_TEST)

#Gives the Structure 
str(ECG_TEST)
#Gies the summary of the test set
summary(ECG_TEST)

#defining the normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#normalizing the test set
ECG_TEST_N = as.data.frame(lapply(ECG_TEST[,c(2:97)], normalize))

#Viewing the normalized test set for kNN model
View(ECG_TEST_N)

#loading the train set
ECG_TRAIN = readxl::read_excel("/Users/lokeshpalacharla/Library/Mobile Documents/com~apple~CloudDocs/NEU/Classes/Summer 2020/Predictive Analytics/Week 1/ECG200/ECG200_TRAIN.xlsx",col_names = FALSE)

#getting the dimensions of the train set
dim(ECG_TRAIN)

#Viewing the train set
View(ECG_TRAIN)

#Gives the Structure 
str(ECG_TRAIN)
#Gies the summary of the test set
summary(ECG_TRAIN)

#normalizing the train set
ECG_TRAIN_N = as.data.frame(lapply(ECG_TRAIN[,c(2:97)], normalize))

#Viewing the normalized train set for kNN model
View(ECG_TRAIN_N)

#Creating test labels 
ECG_TEST_LABELS =  ECG_TEST[, 1]

View(ECG_TEST_LABELS)

#Creating train lables
ECG_TRAIN_LABELS = ECG_TRAIN[, 1]

View(ECG_TRAIN_LABELS)

#Changing the training lables into Vector
ECG_TRAIN_LABELS_V=unlist(ECG_TRAIN_LABELS)

View(ECG_TRAIN_LABELS_V)

#Applying kNN model for K = 3 and p = 0.5
ECG_PREDS = knnVCN(ECG_TRAIN_N, ECG_TRAIN_LABELS_V, ECG_TEST_N, K = 3, ShowObs=F,method = "minkowski",p =0.5)
View(ECG_PREDS)

ECG_PREDS_V=unlist(ECG_PREDS)

table(ECG_TRAIN_LABELS_V,ECG_PREDS_V)

ConfMatrix = confusionMatrix(factor(ECG_PREDS_V, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)),
      factor(ECG_TRAIN_LABELS_V, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)))


ConfMatrix = confusionMatrix(as.factor(ECG_TRAIN_LABELS_V),as.factor(ECG_PREDS_V))

ConfMatrix = confusionMatrix(factor(ECG_PREDS_V, levels = -1:1),
                             factor(ECG_TRAIN_LABELS_V, levels = -1:1))

ConfMatrix #Accuracy : 0.59 


#Applying kNN model for K = 5 and p = 1
ECG_PREDS_1 = knnVCN(ECG_TRAIN_N, ECG_TRAIN_LABELS_V, ECG_TEST_N, K = 5, ShowObs=F,method = "minkowski",p =1)

View(ECG_PREDS_1)

ECG_PREDS_V_1=unlist(ECG_PREDS_1)

table(ECG_TRAIN_LABELS_V,ECG_PREDS_V_1)

ConfMatrix_1 = confusionMatrix(factor(ECG_PREDS_V_1, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)),
                             factor(ECG_TRAIN_LABELS_V, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)))

ConfMatrix_1 #Accuracy : 0.57   




#Applying kNN model for K = 11 and p = 2
ECG_PREDS_2 = knnVCN(ECG_TRAIN_N, ECG_TRAIN_LABELS_V, ECG_TEST_N, K = 11, ShowObs=F,method = "minkowski",p =2)

View(ECG_PREDS_2)

ECG_PREDS_V_2=unlist(ECG_PREDS_2)

table(ECG_TRAIN_LABELS_V,ECG_PREDS_V_2)

ConfMatrix_2 = confusionMatrix(factor(ECG_PREDS_V_2, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)),
                               factor(ECG_TRAIN_LABELS_V, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)))

ConfMatrix_2 #Accuracy : 0.59  

#Applying kNN model for K = 11 and p = 2
ECG_PREDS_3 = knnVCN(ECG_TRAIN_N, ECG_TRAIN_LABELS_V, ECG_TEST_N, K = 12, ShowObs=F,method = "minkowski",p =4)

View(ECG_PREDS_3)

ECG_PREDS_V_3=unlist(ECG_PREDS_3)

table(ECG_TRAIN_LABELS_V,ECG_PREDS_V_3)

ConfMatrix_3 = confusionMatrix(factor(ECG_PREDS_V_3, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)),
                               factor(ECG_TRAIN_LABELS_V, levels = min(ECG_TRAIN_LABELS_V):max(ECG_TRAIN_LABELS_V)))

ConfMatrix_3 #Accuracy : 0.57 
