import math
import sys

import numpy as np
import matplotlib.pyplot as plt

SAFE = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
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

    xi = [float(v) for v in eval(params['nodes_x'], {'__builtins__': {}}, SAFE)]
    yi = [float(v) for v in eval(params['nodes_y'], {'__builtins__': {}}, SAFE)]
    xstar = float(eval(params['xstar'], {'__builtins__': {}}, SAFE))
    f = make_func(params['func']) if 'func' in params else None
    return xi, yi, xstar, f


# Метод прогонки для трёхдиагональной системы
def solve_tridiagonal(a, b, c, d):
    n = len(b)
    if n == 0:
        return []
    cp = [0.0] * n
    dp = [0.0] * n
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise ZeroDivisionError(f"Деление на ~0 в прогонке (i={i})")
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        cp[i] = c[i] / denom if i < n - 1 else 0.0
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# Построение кубического сплайна
def compute_spline(xi, yi):
    N = len(xi) - 1
    if N < 2:
        raise ValueError("Нужно минимум 3 точки для кубического сплайна")

    h = [xi[i + 1] - xi[i] for i in range(N)]

    m = N - 1
    la = [0.0] * m
    lb = [0.0] * m
    lc = [0.0] * m
    ld = [0.0] * m

    for k in range(m):
        i = k + 1
        la[k] = h[i - 1]
        lb[k] = 2.0 * (h[i - 1] + h[i])
        lc[k] = h[i] if k < m - 1 else 0.0
        ld[k] = 3.0 * ((yi[i + 1] - yi[i]) / h[i]
                       - (yi[i] - yi[i - 1]) / h[i - 1])

    c_internal = solve_tridiagonal(la, lb, lc, ld)
    c = [0.0] + c_internal + [0.0]

    a = list(yi[:N])
    b = [0.0] * N
    d = [0.0] * N
    for i in range(N):
        b[i] = (yi[i + 1] - yi[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a, b, c[:N], d, h


# Вычисление сплайна
def find_interval(xi, x):
    N = len(xi) - 1
    if x <= xi[0]:
        return 0
    if x >= xi[-1]:
        return N - 1
    for i in range(N):
        if xi[i] <= x <= xi[i + 1]:
            return i
    return N - 1


def spline_value(xi, a, b, c, d, x):
    i = find_interval(xi, x)
    t = x - xi[i]
    return i, a[i] + b[i] * t + c[i] * t * t + d[i] * t * t * t


def spline_S2(xi, c, d, x):
    i = find_interval(xi, x)
    t = x - xi[i]
    return 2.0 * c[i] + 6.0 * d[i] * t


# Форматирование
def fmt(x, decimals=8):
    if abs(x) < 10 ** (-decimals + 1):
        return f"{0:.{decimals}f}"
    return f"{x:.{decimals}f}"


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "lab32"
    xi, yi, xstar, f = read_input(filename)
    N = len(xi) - 1

    print("Лабораторная работа 3.2: Кубический интерполяционный сплайн")

    print(f"\n1) Исходные данные: {len(xi)} узлов, {N} интервалов")
    print(f"{'i':>2} | {'xi':>14} | {'yi':>14}")
    for i in range(len(xi)):
        print(f"{i:>2} | {fmt(xi[i], 6):>14} | {fmt(yi[i], 6):>14}")
    print(f"\nТочка: x* = {xstar}")

    a, b, c, d, h = compute_spline(xi, yi)

    print("\n2) Коэффициенты кубического сплайна")
    print(f"{'i':>2} | {'интервал':>20} | {'a_i':>12} | {'b_i':>12} | {'c_i':>12} | {'d_i':>12}")
    for i in range(N):
        interval = f"[{xi[i]:.4g}, {xi[i+1]:.4g}]"
        print(f"{i:>2} | {interval:>20} | {fmt(a[i],6):>12} | {fmt(b[i],6):>12} "
              f"| {fmt(c[i],6):>12} | {fmt(d[i],6):>12}")

    print("\nФормула на интервале i:")
    print("   S(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3")

    # 3. Значение в x*
    i_star, s_star = spline_value(xi, a, b, c, d, xstar)
    print("\n3) Значение сплайна в точке x*")
    print(f"x* = {xstar}")
    print(f"Интервал: i = {i_star} = [{xi[i_star]}, {xi[i_star+1]}]")
    print(f"S(x*) = {fmt(s_star, 12)}")
    if f is not None:
        exact = f(xstar)
        print(f"Точное f(x*) = {fmt(exact, 12)}")
        print(f"|S(x*) - f(x*)| = {abs(s_star - exact):.6e}")

    print("\n4) Проверка прохождения через исходные узлы")
    print(f"{'i':>2} | {'xi':>14} | {'yi':>14} | {'S(xi)':>16} | {'|S(xi) - yi|':>14}")
    max_err = 0.0
    for i in range(len(xi)):
        _, s_xi = spline_value(xi, a, b, c, d, xi[i])
        err = abs(s_xi - yi[i])
        max_err = max(max_err, err)
        print(f"{i:>2} | {fmt(xi[i],6):>14} | {fmt(yi[i],6):>14} "
              f"| {fmt(s_xi,8):>16} | {err:.3e}")
    print(f"\nМаксимальное отклонение: {max_err:.3e}  "
          f"({'OK' if max_err < 1e-8 else 'ВНИМАНИЕ'})")

    print("\n5) Проверка граничных условий")
    s2_left = 2.0 * c[0]
    s2_right = 2.0 * c[N-1] + 6.0 * d[N-1] * h[N-1]
    print(f"S''(x_0) = 2*c_0 = {fmt(s2_left, 10)}")
    print(f"S''(x_N) = 2*c_{N-1} + 6*d_{N-1}*h_{N-1} = {fmt(s2_right, 10)}")
    print(f"|S''(x_0)| = {abs(s2_left):.3e},  |S''(x_N)| = {abs(s2_right):.3e}")
    print(f"{'OK' if abs(s2_left) < 1e-8 and abs(s2_right) < 1e-8 else 'ВНИМАНИЕ'}")

    pad = 0.05 * (xi[-1] - xi[0])
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(N):
        x_seg = np.linspace(xi[i], xi[i+1], 60)
        y_seg = [a[i] + b[i]*(t - xi[i]) + c[i]*(t - xi[i])**2 + d[i]*(t - xi[i])**3
                 for t in x_seg]
        ax.plot(x_seg, y_seg, '-', color=colors[i], linewidth=2.0)
    ax.plot([], [], 'k-', linewidth=2, label='интервалы сплайна S_i(x)')

    ax.plot(xi, yi, 'ko', markersize=8, label='узлы (x_i, y_i)')
    ax.plot([xstar], [s_star], 'r*', markersize=16,
            label=f'x* = {xstar:.4g}, S(x*) = {s_star:.4g}')

    if f is not None:
        x_dense = np.linspace(xi[0] - pad, xi[-1] + pad, 400)
        y_exact = [f(t) for t in x_dense]
        ax.plot(x_dense, y_exact, 'k:', linewidth=1.2, alpha=0.6, label='f(x) — точная функция')

    y_plot_vals = list(yi) + [s_star]
    ylo, yhi = min(y_plot_vals), max(y_plot_vals)
    ypad = 0.15 * (yhi - ylo) if yhi > ylo else 1.0
    ax.set_ylim(ylo - ypad, yhi + ypad)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Кубический интерполяционный сплайн')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spline.png', dpi=120)
    plt.show()
    print("\nГрафик сохранён в spline.png")


if __name__ == "__main__":
    main()