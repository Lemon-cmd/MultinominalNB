# Multinominal Naive Bayes Classifier C++ #          

***Please take a look at the given dataset to understand how it works*** !              

For using a different dataset, please consider using the **last column in the data as the test feature or Y**. If not, then you will have to change up the code.       

# How does it works?

The Classifier estimates the conditional probability of a particular feature given a class as the relative frequency of X(feature) belonging to class(c). The variation takes into account the number of occurrences of term X in training documents from class (c),including multiple occurrences.

# Dataset Tips      
It is up to you to clean your data. There are many ways people separate different columns in their dataset. Please simply update the load method to do the splitting based on your **targeted dataset**.  
   
In this example, I am using the adult data set given by UCI [https://archive.ics.uci.edu/ml/datasets/Adult.]     

It has about 40k entries and 15 features. I removed features that contained continuous values. You can see what columns I removed in the removedXs vector in main.    

# Please don't use features with only 1 and 0 or continous values! #             
  - This is not a boolean multinominal NB classifier. It will be up on a different repository!     
  - This is not a Guassian NB classifier. [https://github.com/Lemon-cmd/GuassianNB/tree/master]    
        
# How to use the NB #    
  - Simply run the bash script to build and run  
  - If you want to target a different trainning and test set, simply go into the main function and change the ***loadTrainD()*** and ***loadTestD()*** methods to target different files.   
  - If you want to modify the code for your own needs, go ahead.
       
### Have fun! ###
