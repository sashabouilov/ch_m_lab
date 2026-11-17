import math
import sys

import numpy as np
import matplotlib.pyplot as plt

SAFE = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt, 'abs': abs,
    'pi': math.pi, 'e': math.e,
}


def make_func(expr):
    code = compile(expr, '<expr>', 'eval')

    def f(x):
        SAFE['x'] = x
        return float(eval(code, {'__builtins__': {}}, SAFE))

    return f


def read_input(filename):
    params = {}
    with open(filename, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            if '=' in ln:
                k, v = ln.split('=', 1)
                params[k.strip()] = v.strip()

    x = np.array([float(v) for v in eval(params['x'], {'__builtins__': {}}, SAFE)])
    if 'y' in params:
        y = np.array([float(v) for v in eval(params['y'], {'__builtins__': {}}, SAFE)])
    else:
        f = make_func(params['func'])
        y = np.array([f(t) for t in x])

    return x, y


# LU-разложение (из лабы 1) с частичным выбором главного элемента
def lu_decompose(A):
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("Матрица вырождена")
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


def solve_lu(L, U, P, b):
    Pb = P @ b
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - sum(L[i, j] * y[j] for j in range(i))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


# МНК: нормальная система X^T X · a = X^T y
def least_squares_fit(x, y, degree):
    N = len(x)
    n = degree

    S = [float(np.sum(x**k)) for k in range(2 * n + 1)]
    T = [float(np.sum(y * x**k)) for k in range(n + 1)]

    M = np.array([[S[i + j] for j in range(n + 1)] for i in range(n + 1)])
    rhs = np.array(T, dtype=float)

    L, U, P, _ = lu_decompose(M)
    a = solve_lu(L, U, P, rhs)
    return a


def poly_value(coeffs, x):
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def mse(coeffs, x, y):
    return float(np.mean((np.array([poly_value(coeffs, t) for t in x]) - y) ** 2))


def format_poly(coeffs):
    parts = []
    for k, c in enumerate(coeffs):
        if abs(c) < 1e-14:
            continue
        if k == 0:
            parts.append(f"{c:+.6g}")
        elif k == 1:
            parts.append(f"{c:+.6g}·x")
        else:
            parts.append(f"{c:+.6g}·x^{k}")
    if not parts:
        return "0"
    return " ".join(parts)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "lab33"
    x, y = read_input(filename)

    print("Лабораторная работа 3.3: Метод наименьших квадратов")

    print(f"\nИсходные данные: {len(x)} точек")
    print(f"{'j':>2} | {'x_j':>12} | {'y_j':>12}")
    for j in range(len(x)):
        print(f"{j:>2} | {x[j]:>12.6f} | {y[j]:>12.6f}")

    results = {}
    for degree in (1, 2, 3):
        coeffs = least_squares_fit(x, y, degree)
        err = mse(coeffs, x, y)
        results[degree] = (coeffs, err)

    for degree in (1, 2, 3):
        coeffs, err = results[degree]
        print(f"\nМногочлен степени {degree}")
        print("Коэффициенты (a_0, a_1, ..., a_%d):" % degree)
        for k, c in enumerate(coeffs):
            print(f"  a_{k} = {c:+.10g}")
        print(f"\nP_{degree}(x) = {format_poly(coeffs)}")
        print(f"\nMSE_{degree} = {err:.6e}")

    print("\nСравнение")
    print(f"{'Степень':>8} | {'MSE':>16} |")
    for degree in (1, 2, 3):
        print(f"{degree:>8} | {results[degree][1]:>16.6e}")

    best = min(results, key=lambda d: results[d][1])
    print(f"\nЛучший многочлен: P_{best}(x)  (MSE = {results[best][1]:.6e})")

    x_min, x_max = x.min(), x.max()
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.5
    x_plot = np.linspace(x_min - pad, x_max + pad, 400)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ko', markersize=9, label='исходные точки', zorder=5)

    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    for degree in (1, 2, 3):
        coeffs, _ = results[degree]
        y_plot = [poly_value(coeffs, t) for t in x_plot]
        plt.plot(x_plot, y_plot, color=colors[degree], linewidth=2,
                 label=f'P_{degree}(x)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('МНК: приближение многочленами степени 1, 2, 3')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mse.png', dpi=120)
    plt.show()
    print("\nГрафик сохранён в mse.png")


if __name__ == "__main__":
    main()