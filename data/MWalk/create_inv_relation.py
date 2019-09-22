import os
import sys

f_in = file(sys.argv[1])
f_out = file(sys.argv[2],'w')

for line in f_in:
	items = line.strip().split('\t')
	src = items[0]
	rel = items[1]
	tgt = items[2]
	
	f_out.write(src + '\t' + rel + '\t' + tgt + '\n')
	f_out.write(tgt + '\t' + '_'+rel + '\t' + src + '\n')	 
f_out.close()
f_in.close()
