import numpy as np

matrix = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])

print("Матрица:\n", matrix)

b = np.linalg.inv(matrix)

print("Обратная матрица к A:\n", b)

list = [[1],
        [0],
        [1]]
vector = np.array(list)
print("Вектор:")
print(vector)

matrix[0, 2] = 1
matrix[1, 2] = 0
matrix[2, 2] = 1
print("Матрица после замены одного столбца:\n", matrix)

l = b.dot(vector)
print("Результат умножения матрицы на вектор:\n", l)

l[2, 0] = -1
print("В копии вектора l заменим третий элемент на -1:\n", l)

l2 = np.dot(-1, l)
print("Находим l2:\n", l2)

Q = np.eye(3)
print("Единичная матрица:\n", Q)
Q[0, 2] = -1
#Q[1, 2] = 0
#Q[2, 2] = 1

print("Единичная матрица после замены:\n", Q)

multiResult = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

for m in range(len(Q)):
     for n in range(len(b[0])):
             if (m!=2):
                multiResult[m][n] = Q[m][m]*b[m][n]+Q[m][2]*b[2][n]
             else:
                multiResult[m][n] = Q[m][m] * b[m][n]

print("Результат умножения матрицы Q на матрицву b: ")
for res in multiResult:
        print(res)