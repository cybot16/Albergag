import os

path = '../ProcessedData'
c=0

for (dirpath, dirnames, filenames) in os.walk(path):
	for i in dirnames:
		for (dirpath1, dirnames1, filenames1) in os.walk(path+i):
			for j in filenames1:
				c+=1

print c
