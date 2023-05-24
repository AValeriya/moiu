from cmath import inf
import numpy as np

from lab1 import get_inv

A = np.array([[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
c = np.array([1, 1, 0, 0, 0])
x = np.array([0, 0, 1, 3, 2])
B = np.array([3, 4, 5])

def get_basis_matrix(A, B):
    A_b = A[:, B - 1]
    return A_b

def get_basis_vector(c, B):
    c_b = c[B - 1]
    return c_b

def simplex_method(c, A, x, B):
    count, A_b = 0, None
    a_tran = A.transpose()
    while True:
        #1,2(получаем матрицу, обратную к базисной, и вектор базисных компонент)
        if count == 0:
            A_b = get_basis_matrix(A, B)
            A_b_inv = np.linalg.inv(A_b)
        else:
            A_b_inv = get_inv(A_b, A_b_inv, A[k])

        i = 0
        c_b = get_basis_vector(c, B)
        #3,4(находим векторы потенциалов и оценок)
        u_t = np.dot(c_b, A_b_inv)
        delta = np.dot(u_t, A) - c
        #5(проверка текущего плана на оптимальность)
        neg_delta, j0 = None, None
        for i in range(len(delta)):
            #6(поиск отрицательной компоненты в векторе оценок)
            if delta[i] < 0:
                neg_delta = delta[i]
                j0 = i+1
                break
        if neg_delta is None:
            return x, B
        #7,8(находим векторы z и Θ)
        z = np.dot(A_b_inv, a_tran[j0-1])
        theta = np.zeros(len(z))
        for index, item in enumerate(z):
            if item <= 0:
                theta[index] = inf
            else:
                theta[index] = x[B[index]-1] / z[index]
        #9,10(нахождение минимума в Θ и проверка ограниченности целевой функции)
        theta_0 = min(theta)
        if theta_0 == inf:   #целевая функция задачи не ограничена сверху на множестве допустимых планов
            print('stop')
            return None, None

        #11,13(преобразование базисного допустимого плана)
        k = np.argmin(theta)+1
        j_star = B[k-1]
        B[k-1] = j0
        for j, i in enumerate(B):
            x[i - 1] = x[i - 1] - theta_0 * z[j]
        x[j0 - 1] = theta_0
        x[j_star - 1] = 0

if __name__ == '__main__':
    for res in simplex_method(c, A, x, B):
        print(res)