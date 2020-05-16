import sys
totalLines = 0
for line in sys.stdin:
    totalLines += 1
    line = line.strip('\n')
    line = line.split('[')
    line = line[1:]
    print 'Gm'+str(totalLines - 1)+'_1 = ', line
