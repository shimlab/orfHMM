#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import inf
from math import log
import numpy as np
from scipy.stats import nbinom
from collections import defaultdict
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # log func

# In[ ]:


def log_func(number):
    '''
    convert to log form
    number: scalar
    output: scalar
    '''
    if number > 0:
        return log(number)
    else:
        return -inf


# # log-sum-exp

# In[ ]:


def logSumExp(x):
    '''
    HMM Notes p17
    log sum(exp(x1)+exp(x2)+...)
    x: list
    result: scalar
    '''
    m = max(x)
    if m == -inf: # infinity
        return -inf
    else:
        minus_m = np.array(x) - m
        result = m + log(sum(np.exp(minus_m)))
        return result   


# # log likelihood f(x)

# In[ ]:


def lnNB(x, alpha, beta, E):
    '''
    Notes p2
    x: scalar. Realization from NB distribution
    alpha, beta: scalar. Parameters from Gamma distribution
    E: scalar. Normalization factor for n-th sequence
    result: scalar. Log likelihood f(x)
    '''
    r = alpha
    p = beta / (E+beta)
    result = nbinom.logpmf(x, r, p) # x follows NB(alpha, beta/(E+beta))
    return result


# # Viterbi

# In[ ]:


'''
    Calculate Viterbi algorithm given the next codon won't start 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    output: predict state
'''
def start_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output):
    
    # state 1 to state 1
    temp = prob_current[0] +  log(1) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 
    prob_next[0] = temp
    output[i+1][0] = 1
        
    # state 10 and state 11 to state 11     
    log_10_11 = prob_current[9] + log(1) 
    log_11_11 = prob_current[10] + log(1) 
    if log_10_11 > log_11_11:
        output[i+1][10] = 10
    else:
        output[i+1][10] = 11
        
    # find maximum
    temp = max(log_10_11, log_11_11)
    prob_next[10] = temp + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E)
    
    return (prob_next, output)


# In[ ]:


'''
    Calculate Viterbi algorithm given the next codon is one of the start codons 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 probabilites. Indicates next forward algorithm, updating this value
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    next_codon: string. Indicates the specific start codon
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    output: predict state
'''
def start_codon_true_viterbi(i, prob_current, prob_next, trans, next_codon, observed_data, alpha_list, beta_list, E, output):
    
    # state 1 to state 1
    log_1_1 = log_func(1 - trans[next_codon][0] - trans[next_codon][1])
    temp = prob_current[0] + log_1_1  + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 
    prob_next[0] = temp   
    output[i+1][0] = 1
            
    # state 1 to state 2
    log_1_2 = log_func(trans[next_codon][0])
    temp = prob_current[0] + log_1_2 + lnNB(observed_data[i+1], alpha_list[1], beta_list[1], E)
    prob_next[1] = temp
    output[i+1][1] = 1         

    # state 1 and state 11 to state 12 
    log_1_12 = prob_current[0] + log_func(trans[next_codon][1])
    log_11_12 = prob_current[10] + log_func(trans[next_codon][2])
    if log_1_12 > log_11_12:
        output[i+1][11] = 1
    else:
        output[i+1][11] = 11
        
    # find maximum
    temp = max(log_1_12, log_11_12) 
    prob_next[11] = temp + lnNB(observed_data[i+1], alpha_list[11], beta_list[11], E)
    
    # state 10 and state 11 to state 11     
    log_10_11 = prob_current[9] + log(1) 
    log_11_11 = prob_current[10] + log_func(1 - trans[next_codon][2])
    if log_10_11 > log_11_11:
        output[i+1][10] = 10
    else:
        output[i+1][10] = 11
        
    # find maximum
    temp = max(log_10_11, log_11_11)
    prob_next[10] = temp + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E)
    
    return (prob_next, output)


# In[ ]:


'''
    Calculate Viterbi algorithm given the next codon won't stop
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    output: predict state
'''
def stop_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output):
    
    # state 4 and state 7 to state 5
    log_4_5 = prob_current[3] + log(1) 
    log_7_5 = prob_current[6] + log(1)
    if log_4_5 > log_7_5:
        output[i+1][4] = 4
    else:
        output[i+1][4] = 7
        
    # find maximum
    temp = max(log_4_5, log_7_5)
    prob_next[4] = temp + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)
        
    # state 14 and state 17 to state 15
    log_14_15 = prob_current[13] + log(1) 
    log_17_15 = prob_current[16] + log(1) 
    if log_14_15 > log_17_15:
        output[i+1][14] = 14
    else:
        output[i+1][14] = 17
        
    # find maximum
    temp = max(log_14_15, log_17_15)
    prob_next[14] = temp + lnNB(observed_data[i+1], alpha_list[14], beta_list[14], E)
    
    return (prob_next, output)


# In[ ]:


'''
    Calculate forward algorithm given the next codon is one of the stop codons 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    output: predict state
'''
def stop_codon_true_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output):
    
    # state 4 to state 5
    prob_next[4] = prob_current[3] + log(1) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)
    output[i+1][4] = 4
        
     # state 7 to state 8
    prob_next[7] = prob_current[6] + log(1) + lnNB(observed_data[i+1], alpha_list[7], beta_list[7], E)
    output[i+1][7] = 7
    
    # state 14 to 15
    prob_next[14] = prob_current[13] + log(1) + lnNB(observed_data[i+1], alpha_list[14], beta_list[14], E)
    output[i+1][14] = 14
    
    # state 17 to state 18
    prob_next[17] = prob_current[16] + log(1) + lnNB(observed_data[i+1], alpha_list[17], beta_list[17], E)
    output[i+1][17] = 17
    
    return (prob_next, output)


# In[ ]:


'''
    Compute Viterbi algorithm
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    num_sequence: scalar indicates n-th sequence
    output: a list of predicted states
'''
def viterbi(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, num_sequence):

    # Find E, observed_data, RNA_data according the order of sequences
    E = E[num_sequence - 1]
    observed_data = observed_data[num_sequence - 1]
    RNA_data = RNA_data[num_sequence - 1]
    
    
    sequence_length = len(RNA_data) # length of data
    output = np.zeros((sequence_length, 21)) # initialize matrix
    prob_current = np.ones(21) * (-inf) # initialize probability, 21 different states
    prob_current[0] = lnNB(observed_data[0], alpha_list[0], beta_list[0], E)
    
    for i in range(sequence_length - 1):
        start_codon = False
        stop_codon = False
        next_codon = ""
        prob_next = np.ones(21) * (-inf)
        
        # Check start and stop codon
        if i + 3 < sequence_length:
            next_codon = RNA_data[i+1] + RNA_data[i+2] + RNA_data[i+3]  
        if next_codon in trans.keys():
            start_codon = True
        if next_codon in stop_codon_list:
            stop_codon = True
        
        # start codon is false
        if start_codon == False:       
            
            prob_next = start_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[0]
            output = start_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[1]
         
        # start codon is ture
        elif start_codon == True:   
            prob_next = start_codon_true_viterbi(i, prob_current, prob_next, trans, next_codon, observed_data, 
                                           alpha_list, beta_list, E, output)[0]
            output = start_codon_true_viterbi(i, prob_current, prob_next, trans, next_codon, observed_data, 
                                           alpha_list, beta_list, E, output)[1]
        
        # stop codon is ture
        if stop_codon == True: 
            prob_next = stop_codon_true_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[0]
            output = stop_codon_true_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[1]
 
        # stop codon is false
        elif stop_codon == False:
            prob_next = stop_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[0]
            output = stop_codon_false_viterbi(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, output)[1]
        
       
        # transter to next state with probability 1
        sure_to_transit = [1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18]
        for k in sure_to_transit:
            temp = prob_current[k] + log(1) + lnNB(observed_data[i+1], alpha_list[k+1], beta_list[k+1], E)
            prob_next[k+1] = temp
            output[i+1][k+1] = k+1


        # state 20 and state 21 to state 21     
        log_20_21 = prob_current[19] + log(1) 
        log_21_21 = prob_current[20] + log(1) 
        if log_20_21 > log_21_21:
            output[i+1][20] = 20
        else:
            output[i+1][20] = 21
        temp = max(log_20_21, log_21_21)
        prob_next[20] = temp + lnNB(observed_data[i+1], alpha_list[20], beta_list[20], E)
    
        prob_current = prob_next
        
    
    output_list = [list(prob_current).index(max(prob_current))+1]
    start = len(output) - 1
    while start >= 1:
        index = int(output_list[-1] - 1)
        current = int(output[start][index])
        output_list.append(current)
        start -= 1
    
    output_list.reverse()
    return output_list


# In[ ]:


'''
    Compute predicted states for all RNA sequences by combining result from each single sequence
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    output: matrix stores the Viterbi algorithm for multiple RNA sequences
'''
def viterbi_sequence(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list):
    output_list = []
    for i in range(len(RNA_data)):
        output_list.append(viterbi(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, i+1))
    return output_list


# In[ ]:




