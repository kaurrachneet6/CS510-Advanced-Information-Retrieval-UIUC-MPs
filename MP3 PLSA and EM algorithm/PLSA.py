import numpy
import sys
import matplotlib.pyplot

#Returns c(w,d) - Count of each word in each document as a list of dictionaries 
#Returns background_counts c(w|Collection)
def get_counts():
    f = open('data/dblp-small.txt', 'r', encoding="utf8")
    documents = [document.strip() for document in f] #TODO is this correct?
    f.close()
    doc_dict = [] #list of dictionary
    corpus_count = {}#c(w|collection)
    print('Calculating counts...')
    corpus_count['C_Length'] = 0 
    
    for document in documents:
        temp_doc={} #Dictionary for each document

        #Initialize length variables
        temp_doc['D_Length'] = 0

        words = document.split()
        for word in words:

            if word in temp_doc:
                temp_doc[word] += 1
            else:
                temp_doc[word] = 1
            if word in corpus_count: 
                corpus_count[word] += 1
            else:
                corpus_count[word] = 1

            temp_doc['D_Length'] += 1
            corpus_count['C_Length'] +=1

        doc_dict.append(temp_doc)
    print('Finished calculating counts...')
    return doc_dict, corpus_count


def init():
    #N - no. of documents in the corpus
    #K - no. of topics
    print("Initialize joint, ndk ...")
    #Initialize ndk, nwk, joint dist
    for i in range(N):
        temp_list_joint = []
        temp_list_ndk = []
        for k in range(K):
            temp_list_joint.append({})
            temp_list_ndk.append(0)
        joint.append(temp_list_joint)
        ndk.append(temp_list_ndk)

    print("Initialize nwk ...")

    for k in range(K):
        dic = {}
        for word in corpus_count:
            if word is not 'C_Length':
                dic[word] = 0
        nwk.append(dic)
        
    print("Initialize pi, theta ...")
    
    #Initialize pi and theta
    # pi[i, j] = p(Topic j|Document i)
    numpy.random.seed(random_seed) #Setting a random seed
    pi = numpy.random.rand(N, K)
    pi = pi / pi.sum(axis=1)[:,None] #Normalized so that sum = 1 over all topics

    # theta[i, j] = p(Word j|Topic i)
    theta = [] #List of K dictionaries, one for each topic
    for k in range(K):
        dic_k = {} #Dictonary for each topic
        sum_k = 0.
        for word in corpus_count:
            if word is not 'C_Length':
                r_no = numpy.random.rand()
                dic_k[word] = r_no
                sum_k += r_no
        for word in dic_k:
            dic_k[word] /= sum_k #Normalized so that sum = 1 over all words in the vocabulary
        theta.append(dic_k)
    return pi, theta


def do_Maximization():
	#Iteration step
	#Computing n_dk, n_wk and updating pi and theta
	print('Performing M step...')
	get_ndk()
	get_nwk()
	new_pi()
	new_theta()
	return


def get_ndk():
   for i in range(N):
       doc = doc_dict[i]
       for k in range(K):
           sum = 0.
           for word in doc:
               if word is not 'D_Length':
                   cwd = doc[word]
                   sum += cwd * joint[i][k][word]
           ndk[i][k] = sum
   return


def get_nwk():
    for k in range(K):
        for word in nwk[k]:
            nwk[k][word] = 0

    for i in range(N):
        doc = doc_dict[i]
        for k in range(K):
            for word in doc:
                if word is not 'D_Length':
                    cwd = doc[word]
                    nwk[k][word] += cwd * joint[i][k][word]
    return


def new_pi():
	for i in range(N):
		den=0.
		for k in range(K):
			den += ndk[i][k]
		for k in range(K):
			pi[i][k] = ndk[i][k]/den
	return 

def new_theta():
	for k in range(K):
		den=0.
		for word in nwk[k]:
			den += nwk[k][word]
		for word in nwk[k]:
			theta[k][word] = nwk[k][word]/den
	return 


def do_Expectation():
    print('Performing E step...')
    get_joint()
    return

def get_joint():    
    for i in range(N):
        for word in doc_dict[i]:
            if word is not 'D_Length':
                joint_sum = 0
                for k in range(K):
                    joint_sum += (pi[i][k] * theta[k][word]) #Sum of joint over all the topics
                den = (1.- lamda)*joint_sum + (lamda/corpus_count['C_Length']) * corpus_count[word]
                for k in range(K):
                    num = (1.- lamda)*pi[i][k] * theta[k][word]
                    joint[i][k][word] = float(num)/den
                    
    return



def calc_loglihood():
    print("Calculating Log Likelihood...")
    loglihood = 0
    for i in range(N):
        doc = doc_dict[i]
        for word in doc:
            if word is not 'D_Length':
                pwD = corpus_count[word]/corpus_count['C_Length']

                topic_sum = 0
                for k in range(K):
                    topic_sum += pi[i][k] * theta[k][word]
                first_term = lamda * pwD
                second_term = (1. - lamda) * topic_sum

                loglihood += numpy.log2(first_term + second_term) * doc[word]
    return loglihood

def run_plsa():
    iterations = 0
    delta_List = []
    loglihood_List = []

    old_loglihood = calc_loglihood()
    loglihood_List.append(old_loglihood)

    while iterations < 100:
        print('Iteration: ' + str(iterations))

        do_Expectation()
        do_Maximization()	

        new_loglihood = calc_loglihood()
        loglihood_List.append(new_loglihood)
        print('New Log Likelihood: ' + str(new_loglihood))

        delta = (old_loglihood - new_loglihood)/(old_loglihood)
        delta_List.append(delta)
        print('Change in likelihood: ' + str(delta))

        if delta < 0.0001:
            print('Converged!!!')
            #plot(loglihood_List, delta_List)
            return
        else:
            old_loglihood = new_loglihood
        iterations += 1
    print('Finished 100 iterations...')
    # plot(loglihood_List, delta_List)
    return


def plot(likelihood, delta):
	#Iteration with likelihood
	#Itera with relative
	x1 = []
	for i in range(len(likelihood)):
		x1.append(i)
	        
	x2 = []
	for i in range(len(delta)):
		x2.append(i+1)
	        	
	#First plot
	matplotlib.pyplot.plot(x1,likelihood, label='Log Likelihood', linestyle='--', marker='o')
	axes = matplotlib.pyplot.gca()
	matplotlib.pyplot.xlabel('Iteration')
	matplotlib.pyplot.ylabel('Log Likelihood')
	matplotlib.pyplot.title("Log Likelihood\n K = " + str(K) + " Lamda = " + str(lamda) + " Seed = " + str(random_seed))
	matplotlib.pyplot.legend()
	matplotlib.pyplot.savefig("Loglihood"+ str(K) + str(lamda) + str(random_seed) +'.png')
	matplotlib.pyplot.close()



	matplotlib.pyplot.plot(x2,delta, label='Relative Change', linestyle='--', marker='x')
	axes = matplotlib.pyplot.gca()
	matplotlib.pyplot.xlabel('Iteration')
	matplotlib.pyplot.ylabel('Relative Change')
	matplotlib.pyplot.title("Plot for Relative Change")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.savefig("RelChange"+ str(K) + str(lamda) + str(random_seed) +'.png')
	matplotlib.pyplot.close()
	return


def extract_words():
    for k in range(K):
        top_words = sorted(theta[k], key=theta[k].get, reverse=True)[:10]
        print("Top Words for topic " + str(k))
        print (top_words)


if __name__ == '__main__': 

    
    if len(sys.argv) < 2:
    	print("Usage: python3 PLSA.py K_value lamda_Value")
    	sys.exit(1)

    doc_dict, corpus_count = get_counts()
    M = len(corpus_count) - 1 #Size of the vocabulary set
    N = len(doc_dict)

    #Initial parameters given in the problem
    K = int(sys.argv[1])
    lamda = float(sys.argv[2])
    #K=20
    #lamda=0.9
    print('K value=', K)
    print('lambda value = ', lamda)

    random_seed = 100
    joint = []
    ndk = []
    nwk = []
    pi, theta = init()

    run_plsa()
    extract_words()