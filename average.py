import re
import numpy as np
mojiretu=[]
for l in open('a.txt'):
    if "(" in l:
        mojiretu.append(l)
#print(str(mojiretu))

mojiretu = ''.join(mojiretu)
    
#print(mojiretu)
match = re.findall('[-]?[0-9]+.[-]?[0-9]+', mojiretu)
match = np.reshape(match,(-1,3))
#一つあたり63
#print(match)
ans = np.zeros((10000,3))
ans2 = np.zeros((100000,3))

for k in range(10) :
    for i in range(4) :
        for j in range(3) :
            ans[k][j]+=float(match[0+(i + k*4)*63][j])
    print(ans[k]/4.0)
print("******************************")
for k in range(10) :
    for i in range(4) :
        for j in range(3) :
            ans2[k][j]+=float(match[4+(i + k*4)*63][j])
    print(ans2[k]/4.0)


