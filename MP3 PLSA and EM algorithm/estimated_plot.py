#Code for estimates of Log likelihood and Relative Change in log Likelihood plots 
#Problem 3 Part (b) and (c)
import matplotlib.pyplot as plt
import math

x1 = []
for i in range(100):
	x1.append(i)
        
y1 = []
for x in x1:
	y1.append(math.exp(x* -1.))


#First plot
plt.plot(x1, y1, label='Relative Change', linestyle='--', marker='o')
axes = plt.gca()
plt.xlabel('Iteration')
plt.ylabel('Relative Change')
plt.title("Relative Change")
plt.legend()
# plt.show()
plt.savefig("Relative Change Estimate")
plt.close()


y2 = []       
for x in x1:
	y2.append(math.exp(x* -1.)*-1)


#Second plot
plt.plot(x1, y2, label='Log Likelihood', linestyle='--', marker='o')
axes = plt.gca()
# axes.set_ylim([0,0.5])
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title("Log Likelihood")
plt.legend()
# plt.show()
plt.savefig("Log Likelihood Estimate")
plt.close()