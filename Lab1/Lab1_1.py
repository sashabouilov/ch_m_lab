import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# Чтение данных
def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    A = np.array([list(map(float, lines[i + 1].split())) for i in range(n)])
    b = np.array(list(map(float, lines[n + 1].split())))

    if A.shape != (n, n):
        raise ValueError("Матрица A должна быть квадратной")
    if b.shape[0] != n:
        raise ValueError("Длина вектора b должна совпадать с размером матрицы")

    return A, b


# LU-разложение с частичным выбором главного элемента
def lu_decompose(A):
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0

    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))

        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("Матрица вырождена или близка к вырожденной")

        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swaps += 1

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, P, swaps


# Прямая подстановка
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]
    return y


# Обратная подстановка
def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i, j] * x[j]
        x[i] = s / U[i, i]
    return x


# Решение системы и обратная матрица
def solve_lu(L, U, P, b):
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = back_substitution(U, y)
    return x


# Обратная матрица
def inverse_lu(L, U, P):
    n = L.shape[0]
    E = np.eye(n)
    A_inv = np.zeros((n, n))
    for i in range(n):
        A_inv[:, i] = solve_lu(L, U, P, E[:, i])
    return A_inv


# Определитель
def det_lu(U, swaps):
    det = 1.0
    for i in range(U.shape[0]):
        det *= U[i, i]
    if swaps % 2 == 1:
        det = -det
    return det


# Красивый вывод
def print_matrix(name, M):
    print(f"\n{name}:")
    print(M)


# main
def main():
    filename = "matrix_1"
    A, b = read_input(filename)

    n = A.shape[0]
    print(f"Размерность матрицы: n = {n}")
    print_matrix("Матрица A", A)
    print_matrix("Вектор b", b.reshape(-1, 1))

    # LU-разложение
    L, U, P, swaps = lu_decompose(A)

    print_matrix("Матрица L", L)
    print_matrix("Матрица U", U)
    print_matrix("Матрица перестановок P", P)

    # Проверка L·U = P·A
    LU = L @ U
    PA = P @ A
    print_matrix("L · U", LU)
    print_matrix("P · A", PA)
    ok1 = np.allclose(LU, PA)
    print(f"Проверка L·U = P·A: {'OK' if ok1 else 'FAIL'}")

    # Решение системы A·x = b
    x = solve_lu(L, U, P, b)
    print_matrix("Решение системы x", x.reshape(-1, 1))

    Ax = A @ x
    print_matrix("A · x", Ax.reshape(-1, 1))
    ok2 = np.allclose(Ax, b)
    print(f"Проверка A·x = b: {'OK' if ok2 else 'FAIL'}")

    # Обратная матрица
    A_inv = inverse_lu(L, U, P)
    print_matrix("Обратная матрица A^(-1)", A_inv)

    # Определитель
    det = det_lu(U, swaps)
    print(f"\nОпределитель det(A) = {det:.6f}")

    # Проверка A · A^(-1) = E
    AA_inv = A @ A_inv
    print_matrix("A · A^(-1)", AA_inv)
    ok3 = np.allclose(AA_inv, np.eye(n))
    print(f"Проверка A·A^(-1) = E: {'OK' if ok3 else 'FAIL'}")


if __name__ == "__main__":
    main()
