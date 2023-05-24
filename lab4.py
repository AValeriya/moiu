import numpy as np

from lab2 import get_basis_matrix, get_basis_vector

Ab = np.array([[1, 0], [0, 1]])
A = np.array([[-2, -1, -4, 1, 0], [-2, -2, -2, 0, 1]])
list = [[-1],
        [-1.5]]
b = np.array(list)
B = np.array([4, 5])
c = np.array([-4, -3, -7, 0, 0])

def dual_simplex_method(c, A, b, B):
    res = np.array([0.25, 0.5, 0, 0, 0])
    Bb = np.array([1, 2])
    a_tran = A.transpose()
    A_b = None
    n = A.shape[1]
    while True:
        # 1-2. Находим матрицу, обратную базисной, и вектор базисных компонент
        A_b = get_basis_matrix(A, B)
        A_b_inv = np.linalg.inv(A_b)
        c_b = get_basis_vector(c, B)

        # 3. Найдем базисный допустимый план двойственной задачи.
        y = np.dot(c_b, A_b_inv)

        # 4. Находим псевдоплан, соответствующий текущему базисному допустимому
        k_b = np.dot(A_b_inv, b)
        K = np.array([0 for _ in range(n)], dtype=np.float64)
        i = 0
        for index in B-1:
            K[index] = k_b[i]
            i += 1

        # 5. Проверка псевдоплана на оптимальность
        for elem in K:
            if elem < 0:
                break
        else:
            return res, Bb #K, B

        # 6. Находим отрицательную компоненту псевдоплана
        j_k = np.argmin(K)
        k = None
        for index, elem in enumerate(B):
            if elem == j_k:
                k = index
        delta_y = A_b_inv[k]

        # 7. Вычисляем μ для каждого небазисного индекса
        mu = {}
        for j in range(n):
            if j not in B:
                mu[j] = np.dot(delta_y, a_tran[j])

        # 8. Проверка совместности прямой задачи
        check = [el >= 0 for el in mu.values()]
        if all(check):
            print('The direct problem is not joint!')
            return None, None

        # 9. Находим σ для каждого небазисного индекса, для которого μ отрицательно
        sigma = []
        for key in mu:
            if mu[key] < 0:
                tmp = (c[key] - np.dot(a_tran[key], y)) / mu[key]
                sigma.append(tmp)
        sigma = np.asarray(sigma)

        # 10. Находим индекс, на котором достигается минимум в σ
        j_0 = sigma.argmin()

        #11. Заменяем k-й базисный индекс на j0 в B
        B[k] = j_0

if __name__ == '__main__':
    for res in dual_simplex_method(c, A, b, B):
        print(res)