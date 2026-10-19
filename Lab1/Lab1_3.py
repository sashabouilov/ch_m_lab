import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# Чтение данных
def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    eps = float(lines[1])

    A = np.array([list(map(float, lines[i + 2].split())) for i in range(n)])
    b = np.array(list(map(float, lines[n + 2].split())))

    if A.shape != (n, n):
        raise ValueError("Матрица A должна быть квадратной")
    if b.shape[0] != n:
        raise ValueError("Длина вектора b не совпадает с n")

    return n, eps, A, b


# Проверка сходимости (диагональное преобладание)
def check_diagonal_dominance(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off:
            return False, i
    return True, -1


# Метод простых итераций
def simple_iteration(A, b, eps, max_iter=10000):
    n = A.shape[0]

    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]
        beta[i] = b[i] / A[i, i]

    norm_alpha = max(sum(abs(alpha[i, j]) for j in range(n)) for i in range(n))

    x = beta.copy()

    for k in range(1, max_iter + 1):
        x_new = alpha @ x + beta
        diff = np.max(np.abs(x_new - x))
        x = x_new
        if diff < eps:
            return x, k, norm_alpha

    return x, max_iter, norm_alpha


# Метод Зейделя
def seidel(A, b, eps, max_iter=10000):
    n = A.shape[0]

    x = np.zeros(n)

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]

        diff = np.max(np.abs(x - x_old))
        if diff < eps:
            return x, k

    return x, max_iter


# Эталонное решение (только для проверки)
def exact_solution(A, b):
    return np.linalg.solve(A, b)


def main():
    filename = "matrix_3"
    n, eps, A, b = read_input(filename)

    print(f"Размерность: n = {n}")
    print(f"Точность:    eps = {eps}")
    print("\nМатрица A:")
    print(A)
    print("\nВектор b:")
    print(b)

    ok, bad_row = check_diagonal_dominance(A)
    if ok:
        print("\nДиагональное преобладание: OK (достаточное условие сходимости)")
    else:
        print(f"\nВнимание: нет диагонального преобладания в строке {bad_row + 1}")
        print("Сходимость не гарантирована.")

    x_ref = exact_solution(A, b)
    print("\nЭталонное решение (np.linalg.solve):")
    print(x_ref)

    x_si, iter_si, norm_alpha = simple_iteration(A, b, eps)
    print("\nМетод простых итераций")
    print(f"||alpha||_inf = {norm_alpha:.6f}")
    print(f"Число итераций: {iter_si}")
    print("Решение:")
    print(x_si)
    print(f"Погрешность по сравнению с эталоном: "
          f"{np.max(np.abs(x_si - x_ref)):.3e}")

    x_sz, iter_sz = seidel(A, b, eps)
    print("\nМетод Зейделя")
    print(f"Число итераций: {iter_sz}")
    print("Решение:")
    print(x_sz)
    print(f"Погрешность по сравнению с эталоном: "
          f"{np.max(np.abs(x_sz - x_ref)):.3e}")

    print("\nСравнение методов")
    print(f"МПИ:     {iter_si} итераций")
    print(f"Зейдель: {iter_sz} итераций")
    if iter_sz < iter_si:
        print(f"Зейдель быстрее в {iter_si / iter_sz:.2f} раз")
    elif iter_sz > iter_si:
        print(f"МПИ быстрее в {iter_sz / iter_si:.2f} раз")
    else:
        print("Методы дали одинаковое число итераций")


if __name__ == "__main__":
    main()
