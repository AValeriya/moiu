import numpy as np


def test():
    x_h = np.array([2, 3, 0, 0])

    c = np.array([[-8], [-6], [-4], [-6]])

    D = np.matrix([[2, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])

    A = np.matrix([[1, 0, 2, 1], [0, 1, -1, 2]])
    j_b = np.array([1, 2])
    j_b_star = np.array([1, 2])
    x_exp = np.array([1.7, 2.4, 0., 0.3])
    x = lab(x_h, c, D, A, j_b, j_b_star)
    if (x_exp != x).any():
        print("test failed")
    else:
        print(x)
        print("success")


def build_H(A, j_b_star, D, j_b):
    H = np.zeros(shape=(np.size(A, axis=0) + len(j_b_star), np.size(A, axis=0) + len(j_b_star)))

    D_star = np.empty(shape=(len(j_b_star), len(j_b_star)))

    row = 0
    col = 0
    for i in j_b_star:
        for j in j_b_star:
            D_star[row, col] = D[i - 1, j - 1]
            col += 1
        row += 1
        col = 0

    A_b_star = np.matrix(np.empty(shape=(np.size(A, axis=0), len(j_b_star))))
    counter = 0
    for tmp in j_b_star:
        A_b_star[:, counter] = A[:, tmp - 1]
        counter += 1
    # A_b_star=basis_matrix
    H[0:len(j_b_star), 0:len((j_b_star))] = D_star
    H[len(j_b_star):, 0:len(j_b_star)] = A_b_star
    H[0:len(j_b_star), len(j_b_star):] = np.transpose(A_b_star)
    return H


def lab(x_h, c, D, A, j_b, j_b_star):
    # conditions
    # 1)
    if len(j_b) != np.linalg.matrix_rank(A):
        exit("does not fit the conditions")
    # 2)
    A_b = np.matrix(np.zeros(shape=(len(j_b), np.size(A, axis=0))))
    counter = 0
    for i in j_b:
        A_b[:, counter] = A[:, i - 1]
        counter += 1

    if int(np.linalg.det(A_b)) == 0:
        exit(1)
    print(f'Матрица {A_b} - невырожденная')
    # 3)
    # a)
    for tmp in j_b:
        if tmp not in j_b_star:
            exit(1)
    # b)
    c_x = c + np.dot(D, np.transpose(np.matrix(x_h)))

    c_b_h_x = np.empty(len(j_b))
    counter = 0
    for tmp in j_b:
        c_b_h_x[counter] = c_x[tmp - 1]
        counter += 1
    u_h_x = np.dot(-c_b_h_x, np.linalg.inv(A_b))

    delta_h_x = np.dot(u_h_x, A) + np.transpose(c_x)

    for tmp in j_b_star:

        delta_h_x = np.array(delta_h_x).squeeze()

        if np.array(delta_h_x).squeeze()[tmp - 1] < 0:
            exit(1)

    # c)

    H = build_H(A, j_b_star, D, j_b)

    if int(np.linalg.det(H)) == 0:
        exit(1)

    while True:

        # step1

        c_x = c + np.dot(D, np.transpose(np.matrix(x_h)))

        A_b = np.matrix(np.zeros(shape=(len(j_b), np.size(A, axis=0))))
        counter = 0
        for i in j_b:
            A_b[:, counter] = A[:, i - 1]
            counter += 1

        if int(np.linalg.det(A_b)) == 0:
            exit(1)
        c_b_h_x = np.empty(len(j_b))
        counter = 0
        for tmp in j_b:
            c_b_h_x[counter] = c_x[tmp - 1]
            counter += 1
        u_h_x = np.dot(-c_b_h_x, np.linalg.inv(A_b))

        delta_h_x = np.dot(u_h_x, A) + np.transpose(c_x)
        delta_h_x = np.array(delta_h_x).squeeze()

        # step2
        if (delta_h_x >= 0).all():
            return x_h

        # step 3
        j0 = 0

        for idx, comp in enumerate(delta_h_x):
            if comp < 0:
                j0 = idx + 1
                break

        # step 4
        l_h = np.zeros(np.size(D, axis=0))
        l_b_star = np.empty(len(j_b_star))
        l_h[j0 - 1] = 1

        b_star = np.zeros(len(j_b_star) + np.size(A, axis=0))

        for count, idx in enumerate(j_b_star):
            b_star[count] = D[idx - 1, j0 - 1]

        b_star[len(j_b_star):] = np.transpose(A[:, j0 - 1])

        b_star = np.transpose(np.matrix(b_star))
        H = build_H(A, j_b_star, D, j_b)

        inv_H = np.linalg.inv(H)

        b_star = np.array(b_star).squeeze()
        x_cover = (-inv_H).dot(b_star)

        for idx, count in enumerate(j_b_star):
            l_b_star[idx] = x_cover[idx]

        l_h[0:len(l_b_star)] = l_b_star

        # step 5
        tetta = np.empty(len(j_b_star))
        tetta_j0 = 0

        delta = float(l_h.T.dot(D).dot(l_h))
        if delta == 0:
            tetta_j0 = float('infinity')
        elif delta > 0:
            tetta_j0 = np.fabs(delta_h_x[j0 - 1]) / delta

        for j in j_b_star:
            if l_h[j - 1] < 0:
                tetta[j - 1] = -x_h[j - 1] / l_h[j - 1]
            elif l_h[j - 1] >= 0:
                tetta[j - 1] = float('infinity')

        j_star = j0
        tetta0 = tetta_j0
        for i in tetta:
            if tetta0 > i:
                tetta0 = i
                j_star = np.where(tetta == i)[0][0] + 1

        # j_star=np.where(tetta==tetta0)[0][0]+1

        if tetta0 == float('infinity'):
            exit('целевая функция задачи не ограничена снизу н множестве допустимых планов')

        # step 6
        x_h = x_h + tetta0 * l_h

        if j_star == j0:
            j_b_star = np.hstack((j_b_star, j_star))
        elif j_star in j_b_star and j_star not in j_b:
            j_b_star = np.delete(j_b_star, np.where(j_b_star == j_star)[0][0])
        elif j_star in j_b:
            s = np.where(j_b == j_star)[0][0]
            j_plus = [i for i in j_b_star if i not in j_b]
            if j_plus:
                j_plus = j_plus[0]
                if float(np.dot(np.linalg.inv(A_b), A[:, j_plus - 1])[s]) != 0:
                    j_b[np.where(j_b == j_star)[0][0]] = j_plus
                    j_b_star = np.delete(j_b_star, np.where(j_b_star == j_star)[0][0])
                else:
                    j_plus = [i for i in j_b_star if i not in j_b]
                    for m in j_plus:
                        if np.dot(np.linalg.inv(A_b), A[:, m - 1])[s] != 0 or j_b != j_b_star.any():
                            exit(1)
                    j_b[np.where(j_b == j_star)[0][0]] = j0
                    j_b_star[np.where(j_b_star == j_star)[0][0]] = j0


if __name__ == '__main__':
    test()