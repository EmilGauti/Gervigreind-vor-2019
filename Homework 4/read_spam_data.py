import numpy as np

def read_matrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
   
    # Bonus question: Why should these features be removed?
    matrix=np.delete(matrix, [1209,860], axis=1)
    tokens.pop(1209)
    tokens.pop(860)

    return matrix, tokens, np.array(Y)
