
def load(filename):
    y=[]
    with open(filename,'r') as f:
        for line in f:
            row = line.strip()
            if row is not '':
                y.append(float(row))
    return y

# Test
#y=load('observations.txt')
#print (y)
