import matplotlib.pyplot as plt
from ddpgTrials import *
from gmTrials import *
def runningAvg(mylist, windowSize = 10):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):

        cumsum.append(cumsum[i-1] + x)
        if i < windowSize:
            moving_ave = cumsum[i] / (i)
        else:
            moving_ave = (cumsum[i] - cumsum[i-windowSize])/windowSize
        moving_aves.append(moving_ave)

    return moving_aves

def averageVector(vectors):
    vecNum = len(vectors)
    average = []
    min = len(vectors[0])
    for vec in vectors:
        if min > len(vec): min = len(vec)

    for i in range(min):
        sum = .0
        for j in range(vecNum):
            sum += vectors[j][i]
        average.append( sum / vecNum )

    return average

plt.plot(runningAvg(averageVector([Gm0_1, Gm0_2, Gm0_3, Gm0_4]), 20), color='b', label = 'gm0')

plt.legend(bbox_to_anchor=(0.95,0.2), loc=1, borderaxespad=0.)
plt.xlabel('Number of training epochs')
plt.ylabel('Average game score')

# plt.savefig("img/Qbert_nature.png",bbox_inches='tight')

plt.show()
