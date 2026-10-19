import numpy as np

# Чтение файла
def read_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    eps = float(lines[1])

    A = np.array([list(map(float, lines[i + 2].split())) for i in range(n)])

    if A.shape != (n, n):
        raise ValueError("Матрица A должна быть квадратной")

    return n, eps, A


# Поиск максимального по модулю внедиагонального элемента
def find_max_offdiag(A):
    n = A.shape[0]
    p, q = 0, 1
    max_val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return p, q, max_val


# Вычисление параметров вращения (c = cos φ, s = sin φ)
def rotation_angle(A, p, q):
    if abs(A[p, q]) < 1e-300:
        return 1.0, 0.0

    theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
    if theta >= 0:
        t = 1.0 / (theta + np.sqrt(theta * theta + 1.0))
    else:
        t = -1.0 / (-theta + np.sqrt(theta * theta + 1.0))

    c = 1.0 / np.sqrt(t * t + 1.0)
    s = t * c
    return c, s


# Обновление матрицы
def apply_rotation(A, p, q, c, s):
    n = A.shape[0]

    app = A[p, p]
    aqq = A[q, q]
    apq = A[p, q]

    A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
    A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
    A[p, q] = 0.0
    A[q, p] = 0.0

    for i in range(n):
        if i == p or i == q:
            continue
        aip = A[i, p]
        aiq = A[i, q]
        A[i, p] = c * aip - s * aiq
        A[p, i] = A[i, p]
        A[i, q] = s * aip + c * aiq
        A[q, i] = A[i, q]


# Накопление матрицы собственных векторов: V ← V · U_{pq}
def accumulate_vectors(V, p, q, c, s):
    n = V.shape[0]
    for i in range(n):
        vip = V[i, p]
        viq = V[i, q]
        V[i, p] = c * vip - s * viq
        V[i, q] = s * vip + c * viq


# Метод вращений (Якоби)
def rotation_method(A, eps, max_iter=10000):
    n = A.shape[0]
    A = A.copy().astype(float)
    V = np.eye(n)
    history = []

    k = 0
    for k in range(1, max_iter + 1):
        p, q, max_val = find_max_offdiag(A)
        history.append(max_val)

        if max_val < eps:
            break

        c, s = rotation_angle(A, p, q)
        apply_rotation(A, p, q, c, s)
        accumulate_vectors(V, p, q, c, s)

    eigenvalues = np.diag(A).copy()
    return eigenvalues, V, k, history


# Проверка A · V = V · Λ
def check_AV_VL(A, eigenvalues, eigenvectors):
    Lambda = np.diag(eigenvalues)
    AV = A @ eigenvectors
    VL = eigenvectors @ Lambda
    diff = np.max(np.abs(AV - VL))
    return AV, VL, diff


def main():
    filename = "matrix_4"
    n, eps, A = read_file(filename)

    print(f"Размерность: n = {n}")
    print(f"Точность:    eps = {eps}")
    print("\nИсходная матрица A:")
    print(A)

    eig_ref = np.sort(np.linalg.eigvalsh(A))

    eigenvalues, eigenvectors, iters, history = rotation_method(A, eps)

    order = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[order]
    eigenvectors_sorted = eigenvectors[:, order]

    print("\nРезультат метода вращений")
    print(f"Число итераций: {iters}")

    print("\nСобственные значения:")
    print(eigenvalues_sorted)

    print("\nЭталонные собственные значения (np.linalg.eigvalsh):")
    print(eig_ref)

    print("\nПогрешность по собственным значениям:")
    print(np.max(np.abs(eigenvalues_sorted - eig_ref)))

    print("\nСобственные векторы (по столбцам):")
    print(eigenvectors_sorted)

    print("\nПроверка A · V = V · Λ")

    AV, VL, diff = check_AV_VL(A, eigenvalues_sorted, eigenvectors_sorted)

    print("\nМатрица A · V:")
    print(AV)

    print("\nМатрица V · Λ:")
    print(VL)

    print(f"\n||A·V − V·Λ||_inf = {diff:.3e}")
    if diff < 10 * eps:
        print("Проверка A · V = V · Λ: OK")
    else:
        print("Проверка A · V = V · Λ: FAIL")

    print("\nАнализ сходимости")
    print(f"{'Итерация':>10} | {'max|a_ij|':>15}")
    print("-" * 30)
    for k, val in enumerate(history, start=1):
        print(f"{k:>10} | {val:>15.6e}")

    print(f"\nmax|a_ij| до итераций:  {history[0]:.6e}")
    print(f"max|a_ij| после итераций: {history[-1]:.6e}")
    print(f"Заданная точность:        {eps:.6e}")


if __name__ == "__main__":
    main()
