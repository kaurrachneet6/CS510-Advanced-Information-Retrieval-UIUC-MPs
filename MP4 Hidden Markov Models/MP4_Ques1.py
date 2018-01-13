import numpy as np
import pandas as pd

#States
S=['V', 'N']

#Words
W=['print', 'line', 'hello', 'world']

#State transition Probability matrix
A=pd.DataFrame(np.array([[0.6, 0.4], [0.5, 0.5]]), index = S, columns = S)

#Initial Probabilities of starting from a particular state
pi = pd. Series(np.array([0.5, 0.5]), index = S)

#Emission probabilities to generating a word when in a particular state
B = pd.DataFrame(np.array([[ 0.2, 0.1, 0.6, 0.1], [0.1, 0.2, 0.1, 0.6]]), index = S, columns = W)

#Alpha - Forward Probabilities - dataframe with zeros to be filled, index = states, columns = words of the sentence observed 
alpha = pd.DataFrame(0.,index = S, columns = list(range(len(W))))

#Beta - Backward Probabilities - dataframe with zeros to be filled, index = states, columns = words of the sentence observed 
beta = pd.DataFrame(0.,index = S, columns = list(range(len(W))))

#Ques 1 Part (a)
#Implementing the forward algorithm 

def forward_algo(): 
    #Filling the first coloumn of alpha at t = 0
    t_initial=0
    for i in S:
        alpha.ix[i, t_initial]= pi[i]*B.ix[i, W[t_initial]]
    #Filling the last 3 columns 
    for t in range(1, len(W)):
        for i in S:
            for j in S:
                alpha.ix[i, t] += B.ix[i, W[t]]*alpha.ix[j, t-1]*A.ix[j,i] #State S[i] and Word W[t]
    forward=0.    #P('print line hello world')
    for i in S:
        forward+=alpha.ix[i,len(W)-1]
    return forward   #Data Likelihood Probability


#Computing the P('print line hello world'| lambda) and Alpha values
print ('\nQuestion 1 Part (a) Forward Algorithm')
forward_prob = forward_algo()
print ('Alpha =') #Forward probabilities
print(alpha)
print ('\nP("print line hello world") = P(X|lambda) = ', forward_prob, '(Using Forward Algorithm)') #Data Likelihood using Forward Algorithm


#Printing the alpha values 
print ('\nAlpha values are as follows:')
for i in range(len(W)):
    for k in S:
        print ('Alpha_',i+1,'(', k,')= ', alpha.ix[k,i])


#Ques 1 Part (b)
#Implementing the backward algorithm 

def backward_algo(): 
    #Filling the last coloumn of beta at t = len(W)-1
    t_final=len(W)-1
    for i in S:
        beta.ix[i, t_final]= 1.
    #Filling the first 3 columns 
    for t in reversed(range(len(W)-1)):
        for i in S:
            for j in S:
                beta.ix[i, t] += B.ix[j, W[t+1]]*beta.ix[j, t+1]*A.ix[i,j] #State S[i] and Word W[t]
    backward=0.    #P('print line hello world')
    for i in S:
        backward+=alpha.ix[i,0]*beta.ix[i,0]
    return backward     #Data Likelihood Probability


#Computing the P('print line hello world'| lambda) and beta values
print ('\n\nQuestion 1 Part (b) Backward Algorithm')
backward_prob = backward_algo()
print ('Beta =') #Backward probabilities
print(beta)
print ('\nP("print line hello world") = P(X|lambda) = ', backward_prob, '(Using Backward Algorithm)') 
#Data Likelihood using Backward Algorithm


#Printing the beta values 
print ('\nBeta values are as follows:')
for i in range(len(W)):
    for k in S:
        print ('Beta_',i+1,'(', k,')= ', beta.ix[k,i])




