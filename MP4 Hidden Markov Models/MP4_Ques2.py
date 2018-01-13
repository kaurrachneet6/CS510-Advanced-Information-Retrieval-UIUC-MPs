import numpy as np
import json
import pandas as pd


#Training data set to estimate parameters
with open("data/train.json") as fin:
    data = json.load(fin)

#POS tags
tags=['RB','NN','CC','VBN','JJ','IN','VBZ','DT','NNS','NNP']

#Initializing Start Probabilities - pi as an empty series to be filled
pi_temp=pd.Series(0.,index=tags)

#Initializing Transition Probabilities - A as an empty dataframe to be filled - Prob(t_j|t_i)
A_temp=pd.DataFrame(0., index=tags, columns = tags)

#Emission probabilities to generating a word when in a particular state - Prob(w_j|t_i)
B_temp = pd.DataFrame(index = tags)


'''
Question 2 Part (a)
Computing the Start Prob, Trnasition Prob. and Emission Prob.
'''

#Computing the start probabilities series pi
for i in range(len(data)):
    pi_temp[data[i]['pos_tags'][0]]+=1. #No. of times the first tag of the sentence is the given tag
#Normalizing by the total no. of sentences in the dataset
pi=pi_temp/pi_temp.sum()                 #Series with Start Prob.


#Computing the transition probabilities matrix A
for d in range(len(data)):
    for i, elem in enumerate(data[d]['pos_tags']):
        try:
            A_temp.ix[elem][data[d]['pos_tags'][i+1]]+=1
        except:
            pass
A=(A_temp.T/A_temp.sum(axis=1)).T #Data Frame with transition Probabilities 

#Computing the emission probabilities matrix B
for d in range(len(data)):
    for i, elem in enumerate(data[d]['pos_tags']):
        try:
            B_temp.ix[elem][data[d]['words'][i]]+=1         
        except:
            B_temp[data[d]['words'][i]]=0.
            B_temp.ix[elem][data[d]['words'][i]]+=1
B=(B_temp.T/B_temp.sum(axis=1)).T #Data Frame with emission Probabilities


print ('\nQuestion 2 Part (a):\n')
#Printing the start Probabilities
print ('The start Probabilities for each of the POS tags is as follows:')
print (pi)


#Series for top 10 words for each POS tag with highest output Prob.
top10 = pd.Series(0., index=tags)
#Printing the top 10 words for each tag based on emission Probabilities
print ('\nTop 10 words with the highest output probabilities for each POS tags are as follows:\n')
for tag in tags:
    top10[tag]=list(B.ix[tag].nlargest(10).index.values)
    print (tag,':', top10[tag])

'''
Question 2 Part (b)
Viterbi Algorithm
'''
#Testing data set to computes most likely POS tags
f = open('data/test_0')
test0=f.read().splitlines() #Splitting each line as a separate sentence
f.close()
#Splitting the test0 data in a list of lists with each word of the sentence separated 
test0_data=[]
for i in range(len(test0)):
    test0_data.append(test0[i].split())


#Defining the Viterbi Algorithm for the given sentence based on pi, A and B computed in Ques 2, Part(a)
def Viterbi(s):
    length=len(s)
    log_VP=[None]*length
    #log(Viterbi path prob.) - log(Prob. of best path seen so far) - same size as no. of words in the sentence
    q = [None]*length #Best path seen so far - same size as no. of words in the sentence
    
    #Determining the first state based on max prob.
    try:
        prob=np.log(B[s[0]]+np.finfo(float).eps)+np.log(pi) #Using the logarithm sum to be maximized 
    except:
        prob=np.log(pi) #Using Method 2 to deal with unseen words
    log_VP[0] = prob.max()
    q[0] = prob.idxmax()
    
    #Determining the rest of the states based on max prob. 
    for t in range(1, length):      
        try:
            #Optimal path ending in each of the states at time t 
            prob=log_VP[t-1]+np.log(A.ix[q[t-1]])+np.log(B[s[t]]+np.finfo(float).eps)  
            #Using the logarithm sum to be maximized 
        except:
            prob=log_VP[t-1]+np.log(A.ix[q[t-1]]) #Using method 2 to deal with unseen words
        #Max over all states optimal paths until this state
        log_VP[t] = prob.max()
        q[t] = prob.idxmax()        
    return (q)

#Running the Viterbi Algorithm for test_0 file and writing results to a new .txt file
print ('\nQuestion 2 Part (b):\n')
print ('Writing the tagging results for test_0 file using simple Viterbi Algorithm to a new file named \'data/tag_results_test_0_RK\'  \n')
file = open('data/tag_results_test_0_RK','w') 
for index in range(len(test0_data)):
    tag_result = Viterbi(test0_data[index]) #Running the Viterbi Algorithm for test_0 file
    for tag in tag_result:
        file.write(tag+' ')                #Writing the generated tags on a new text file
    file.write('\n')
file.close() 


'''
Question 2 Part (c)
Method 1: Viterbi Algorithm with Laplace smoothing
Method 2: Ignoring the B values for the unseen words in Viterbi Algorithm, i.e. VP = VP*A is maximized rather than VP=VP*A*B
'''
#Testing data set to computes most likely POS tags
f = open('data/test_1')
test1=f.read().splitlines() #Splitting each line as a separate sentence
f.close()
#Splitting the test0 data in a list of lists with each word of the sentence separated 
test1_data=[]
for i in range(len(test1)):
    test1_data.append(test1[i].split())


'''
Implementing Method 2:
Running the Viterbi Algorithm (Using Method 2 of not considering B values in Viterbi algorithm) 
for test_1 file and writing results to a new .txt file
'''
print ('\nQuestion 2 Part (c):\n')
print ('Writing the tagging results for test_1 file using modified Viterbi Algorithm by method 2 to a new file named \'tag_results_test_1_RK_Method2_Modified_Viterbi\' \n')
file = open('data/tag_results_test_1_RK_Method2_Modified_Viterbi','w') 
for index in range(len(test1_data)):
    tag_result = Viterbi(test1_data[index]) #Running the Viterbi Algorithm for test_1 file
    for tag in tag_result:
        file.write(tag+' ')                #Writing the generated tags on a new text file
    file.write('\n')
file.close() 


#Implementing Laplace Smoothing (Since test_1 has unseen words in the training set)

#Smoothing pi values
pi_temp=pi_temp+1
pi=pi_temp/pi_temp.sum()

#Smoothing Transition matrix values
A_temp=A_temp+1
A=(A_temp.T/A_temp.sum(axis=1)).T


#Smoothing Emission Probabilities
for d in range(len(test1_data)):
    for elem in test1_data[d]:
        try:
            B_temp[elem]+=0.        
        except:
            B_temp[elem]=0.            
B_temp=B_temp+1
B=(B_temp.T/B_temp.sum(axis=1)).T


'''
Method 1: Using Laplace Smoothing the Viterbi Algorithm 
Running the Smoothed Viterbi Algorithm (Using Laplace Smoothing) for test_1 file and writing results to a new .txt file
'''
print ('\nQuestion 2 Part (c):\n')
print ('Writing the tagging results for test_1 file using Smoothed Viterbi Algorithm to a new file named \'tag_results_test_1_RK_LaplaceSmoothed_Viterbi\' \n')
file = open('data/tag_results_test_1_RK_LaplaceSmoothed_Viterbi','w') 
for index in range(len(test1_data)):
    tag_result = Viterbi(test1_data[index]) #Running the Viterbi Algorithm for test_1 file
    for tag in tag_result:
        file.write(tag+' ')                #Writing the generated tags on a new text file
    file.write('\n')
file.close() 


'''
Question 2 Part (d)
Accuracy for Viterbi Algorithm
'''

#Computing the accuracy for tag results for the test_0 file
f1 = open('data/test_0_tagging')
f2=open('data/tag_results_test_0_RK')
test0_tags_original=f1.read().splitlines() #Splitting each line as a separate sentence
test0_tags_computed=f2.read().splitlines() #Splitting each line as a separate sentence
f1.close()
f2.close()

#Accuracy for test_0 tagging results using simple Viterbi Algorithm
print ('Question 2 Part (d):\n')
print ('Accuracy results:\n')
accurate=0.  #No. of accurate tags
total=0.    #Total no. of tags
for index in range(len(test0_tags_original)):
    accurate+=sum([1. for i,j in zip(test0_tags_original[index].split(),test0_tags_computed[index].split()) if i==j])
    total+=len(test0_tags_original[index].split())
print ('Accuracy for tagging in test_0 file using Viterbi Algorithm: ', accurate/total)


'''
Computing the accuracy for tag results for the test_1 file using 
Method 1: Laplace Smoothing and Method 2: Neglecting B for unseen words
'''
f1 = open('data/test_1_tagging')
f2=open('data/tag_results_test_1_RK_LaplaceSmoothed_Viterbi')
f3=open('data/tag_results_test_1_RK_Method2_Modified_Viterbi')
test1_tags_original=f1.read().splitlines() #Splitting each line as a separate sentence
test1_tags_computed_Method1=f2.read().splitlines() #Splitting each line as a separate sentence
test1_tags_computed_Method2=f3.read().splitlines() #Splitting each line as a separate sentence
f1.close()
f2.close()
f3.close()


#Accuracy for test_1 tagging results using Laplace smoothed Viterbi Algorithm
accurate1=0.  #No. of accurate tags in Laplace Smoothing
accurate2=0. #No. of accurate tags using Method 2 of ignoring B values in Viterbi when unseen word is encountered
total=0.    #Total no. of tags
for index in range(len(test1_tags_original)):
    accurate1+=sum([1. for i,j in zip(test1_tags_original[index].split(),test1_tags_computed_Method1[index].split()) if i==j])
    accurate2+=sum([1. for i,j in zip(test1_tags_original[index].split(),test1_tags_computed_Method2[index].split()) if i==j])
    total+=len(test1_tags_original[index].split())
print ('Accuracy for tagging in test_1 file using Laplace Smoothed Viterbi Algorithm: ', accurate1/total)
print ('Accuracy for tagging in test_1 file using Modified Viterbi Algorithm where we compute VP=VP*A for unseen words: ', accurate2/total, '\n')




