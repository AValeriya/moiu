import sys

import numpy as np

from lab2 import simplex_method, get_basis_matrix

A = np.array([[1, 1, 1], [2, 2, 2]])
list = [[0],
        [0]]
b = np.array(list)
c = np.array([1, 0, 0])

def get_basis_plan(c, A, b):
    #1(преобразование задачи для соблюдения неотрицательности)
    for index, elem in enumerate(b):
        if elem < 0:
            elem *= -1
            A[index] *= - 1

    m = A.shape[0]
    n = A.shape[1]

    #2(cоставим вспомогательную задачу линейного программирования)
    c_dashed = np.array([0 if i < n else -1 for i in range(n + m)])
    a_dashed = np.append(A, np.eye(2), axis=1)

    #3(построим начальный базисный допустимый план)
    x_dop = np.append(np.array([0 for _ in range(n)]), b)
    B = np.array([n + i for i in range(m)])

    #4(решим вспомогательную задачу основной фазой симплекс-метода)
    x, B = simplex_method(c_dashed, a_dashed, x_dop, B)
    a_tran = a_dashed.transpose()
    a_basis_inv = np.linalg.inv(get_basis_matrix(a_dashed, B))

    #5(проверка условия совместности)
    for i in range(m):
        if x[n + i] != 0:   #несовместна
            print("STOP")
            sys.exit()

    #6(формируем допустимый план задачи)
    x = x[:n]
    while True:
        #7(проверка допустимости текущего базисного плана)
        basis = [j <= n for j in B]
        if all(basis):
            return x, B, A, #b + 1


        #8(находим максимальный индекс искусственной переменной)
        k = np.argmax(B)

        #9(находим векторы l для каждого индекса от 1 до n, которого нет в В)
        l = {}
        for j in range(n):
            if j not in B:
                l[j] = np.dot(a_basis_inv, a_tran[j])

        #10(преобразование множества базисных индексов)
        for j in l:
            if l[j][k] != 0:
                B[k] = j - 1

        # 11(удаление линейно зависимых ограничений)
        for j in l:
            if l[j][k] == 0:
                A = np.delete(A, j - 1, 0)
                a_dashed = np.delete(a_dashed, j, 0)
                b = np.delete(b, j)
                B = np.delete(B, j)
                break

if __name__ == '__main__':
    for res in get_basis_plan(c, A, b):
        print(res)