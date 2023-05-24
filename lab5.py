import numpy as np

def test():
    a = np.array([100, 300, 300])
    b = np.array([300, 200, 200])
    c = np.matrix([[8, 4, 1], [8, 4, 3], [9, 7, 5]])
    x_expected = np.matrix([[0, 0, 100], [0, 200, 100], [300, 0, 0]])
    x=lab(a,b,c)
    if (x_expected != x).any():
        print("test failed")
    else:
        print(x)
        print("test passed successful")

class Node:
    def __init__(self, i: int, j: int, value: int, sign: str):
        self.i = i
        self.j = j
        self.value = value
        self.sign = sign
def lab(aa,bb,cc):
    iter=0
    a = aa#np.array([100, 300, 300])
    b = bb#np.array([300, 200, 200])
    c = cc#np.matrix([[8, 4, 1], [8, 4, 3], [9, 7, 5]])
    #x = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # print(a)
    # print(b)
    # print(c)
    #print(x)
    x = np.zeros((np.size(a), np.size(b)))
    B = []
    i = 0
    j = 0
    a_copy = a.copy()
    b_copy = b.copy()

    # 1 phase
    while i < np.size(a) and j < np.size(b):
        iter+=1
        if a_copy[i] <= b_copy[j]:
            b_copy[j] -= a_copy[i]
            x[i, j] = a_copy[i]
            a_copy[i] = 0
            B.append((i + 1, j + 1))
            i += 1
        elif a_copy[i] > b_copy[j]:
            a_copy[i] -= b_copy[j]
            x[i, j] = b_copy[j]
            b_copy[j] = 0
            B.append((i + 1, j + 1))
            j += 1

    # print(B)
    # print(x)

    # 2 phase
    while True:
        b = []
        i = 0
        A = np.zeros((np.size(c, axis=0) + np.size(c, axis=1), np.size(c, axis=0) + np.size(c, axis=1)))
        for tmp in B:
            u, v = tmp
            u -= 1
            v -= 1
            b.append(c[u, v])
            A[i, u] = 1
            A[i, v + np.size(c, axis=1)] = 1
            i += 1
        A[i, 0] = 1
        b.append(0)
        # print(b)
        # print(A)
        uv = np.linalg.solve(A, b)
        # print("uv",uv)


        def find_base():

            for i in range(np.size(c,axis=0)):
                for j in range(np.size(c,axis=1)):
                    if (i + 1, j + 1) not in B and uv[i] + uv[np.size(c,axis=0) + j] > c[i, j]:
                        print(i + 1, j + 1)
                        base = (i + 1, j + 1)
                        B.append(base)
                        return base
            return 0
        #base = 0
        base = find_base()
        # print(base)
        if base == 0:
            return x
        # print(B)



        B_copy=B.copy()
        m = c.shape[0]
        n = c.shape[1]
        is_deleted = True
        num_in_row = 0
        while is_deleted:
            is_deleted = False
            for k in range(m):
                for i, j in B_copy:
                    if i-1 == k:
                        num_in_row += 1
                if num_in_row == 1:
                    B_copy = [(i, j) for (i, j) in B_copy if i-1 != k]
                    is_deleted = True
                num_in_row = 0

            num_in_col = 0
            for k in range(n):
                for i, j in B_copy:
                    if j-1 == k:
                        num_in_col += 1
                if num_in_col == 1:
                    B_copy = [(i, j) for (i, j) in B_copy if j-1 != k]
                    is_deleted = True
                num_in_col = 0
        # print(B_copy)
        #
        # print(base)



        new_B = [[i, j, False] for (i, j) in B_copy]

        new_base_i, new_base_j = base
        # print(new_base_i)
        next_sign = '+'
        graph = [Node(new_base_i, new_base_j, x[new_base_i-1, new_base_j-1], next_sign)]
        if next_sign == '+':
            next_sign= '-'
        elif next_sign == '-':
            next_sign= '+'
        else:
            raise RuntimeError('Unknown cur_sign')
        #next_sign = get_next_sign(next_sign)
        prev_i = new_base_i
        prev_j = new_base_j

        is_end = False
        while not is_end:
            for k in range(len(new_B)):
                i, j, is_visited = new_B[k]
                if (i == prev_i and j != prev_j) or (j == prev_j and i != prev_i):
                    if is_visited:
                        continue
                    if i == new_base_i and j == new_base_j:
                        is_end = True
                        break
                    graph.append(Node(i, j, x[i-1, j-1], next_sign))
                    if next_sign == '+':
                        next_sign = '-'
                    elif next_sign == '-':
                        next_sign = '+'
                    else:
                        raise RuntimeError('Unknown cur_sign')
                    # next_sign = get_next_sign(next_sign)
                    prev_i = i
                    prev_j = j
                    new_B[k][2] = True
                    break

        # for tmp in graph:
        #     print(tmp.value)
        tetta = graph[1].value
        # print(tetta)
        index = 1
        for i in range(3, len(graph), 2):
            if graph[i].value < tetta:
                tetta = graph[i].value
                index = i

        for n in graph:
            if n.sign == '+':
                n.value += tetta
            else:
                n.value -= tetta
        print()
        # for tmp in graph:
        #     print(tmp.value)

        for n in graph:
            x[n.i-1, n.j-1] = n.value
        print(x)
        # print((graph[index].i, graph[index].j))
        B.remove((graph[index].i, graph[index].j))
        print(B)

if __name__ == '__main__':
    test()
    #x=lab()
    #print(x)