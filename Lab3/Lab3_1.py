import math
import sys

import numpy as np
import matplotlib.pyplot as plt

SAFE = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'ctg': lambda x: 1.0 / math.tan(x),
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'exp': math.exp, 'log': math.log, 'log10': math.log10, 'lg': math.log10,
    'sqrt': math.sqrt, 'abs': abs,
    'pi': math.pi, 'e': math.e,
}


def make_func(expr):
    code = compile(expr, '<expr>', 'eval')

    def f(x):
        SAFE['x'] = x
        return float(eval(code, {'__builtins__': {}}, SAFE))

    return f


def read_input(filename):
    with open(filename, 'r', encoding='utf-8') as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]

    params = {}
    for ln in lines:
        if '=' in ln:
            k, v = ln.split('=', 1)
            params[k.strip()] = v.strip()

    f = make_func(params['func'])
    xi = list(eval(params['nodes'], {'__builtins__': {}}, SAFE))
    xstar = float(eval(params['xstar'], {'__builtins__': {}}, SAFE))

    return f, np.array(xi, dtype=float), xstar


# Многочлен Лагранжа
def lagrange_value(xi, yi, x):
    n = len(xi)
    total = 0.0
    for i in range(n):
        term = yi[i]
        for j in range(n):
            if j != i:
                term *= (x - xi[j]) / (xi[i] - xi[j])
        total += term
    return total


def format_lagrange(xi, yi):
    n = len(xi)
    parts = []
    for i in range(n):
        coeff = yi[i]
        for j in range(n):
            if j != i:
                coeff /= (xi[i] - xi[j])
        factors = [f"(x - {xi[j]:.6g})" for j in range(n) if j != i]
        term = f"({coeff:+.6g})"
        if factors:
            term += " * " + " * ".join(factors)
        parts.append(term)
    return "\n   ".join(parts)


# Таблица разделённых разностей
def divided_differences_table(xi, yi):
    n = len(xi)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = yi[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i][j - 1] - table[i + 1][j - 1]) / (xi[i] - xi[i + j])
    return table


def newton_value(xi, table, x):
    n = len(xi)
    result = table[0][0]
    prod = 1.0
    for j in range(1, n):
        prod *= (x - xi[j - 1])
        result += table[0][j] * prod
    return result


def format_newton(xi, table):
    n = len(xi)
    parts = [f"{table[0][0]:+.6g}"]
    for j in range(1, n):
        factors = [f"(x - {xi[k]:.6g})" for k in range(j)]
        term = f"({table[0][j]:+.6g})"
        if factors:
            term += " * " + " * ".join(factors)
        parts.append(term)
    return "\n   ".join(parts)


def print_divided_table(xi, table):
    n = len(xi)
    header = f"{'i':>2} | {'xi':>12} | {'f[xi]':>12} | " + " | ".join(
        f"{'порядок ' + str(j):>14}" for j in range(1, n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{i:>2} | {xi[i]:>12.6g} | {table[i][0]:>12.6g} |"
        for j in range(1, n):
            if i + j < n:
                row += f" {table[i][j]:>14.6g} |"
            else:
                row += " " * 16 + "|"
        print(row)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "lab31"
    f, xi, xstar = read_input(filename)
    yi = np.array([f(x) for x in xi])
    n = len(xi)

    print("Лабораторная работа 3.1: Интерполяция Лагранжа и Ньютона")

    print("\nУзлы интерполяции:")
    print(f"{'i':>2} | {'xi':>15} | {'yi = f(xi)':>15}")
    for i in range(n):
        print(f"{i:>2} | {xi[i]:>15.8g} | {yi[i]:>15.8g}")
    print(f"\nТочка интерполяции: x* = {xstar:.10g}")

    L = lagrange_value(xi, yi, xstar)
    print("\n1) Многочлен Лагранжа")
    print(f"L_{n-1}(x) =")
    print("   " + format_lagrange(xi, yi))
    print(f"\nL_{n-1}(x*) = {L:.12g}")

    table = divided_differences_table(xi, yi)
    print("\n2) Таблица разделённых разностей")
    print_divided_table(xi, table)

    N = newton_value(xi, table, xstar)
    print("\n3) Многочлен Ньютона")
    print(f"P_{n-1}(x) =")
    print("   " + format_newton(xi, table))
    print(f"\nP_{n-1}(x*) = {N:.12g}")

    exact = f(xstar)
    print("\n4) Сравнение с точным значением")
    print(f"Точное значение f(x*) = {exact:.12g}")
    print(f"\nЛагранж: L(x*) = {L:.12g}")
    print(f"         |L(x*) - f(x*)| = {abs(L - exact):.3e}")
    print(f"\nНьютон:  P(x*) = {N:.12g}")
    print(f"         |P(x*) - f(x*)| = {abs(N - exact):.3e}")
    print(f"\n|L(x*) - P(x*)| = {abs(L - N):.3e}   (≈ 0, так как многочлены совпадают)")

    pad = 0.15 * (xi.max() - xi.min()) if xi.max() > xi.min() else 0.5
    x_plot = np.linspace(xi.min() - pad, xi.max() + pad, 300)
    y_exact_plot = np.array([f(t) for t in x_plot])
    y_L_plot = np.array([lagrange_value(xi, yi, t) for t in x_plot])
    y_N_plot = np.array([newton_value(xi, table, t) for t in x_plot])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_exact_plot, 'k-', linewidth=2, label='f(x) — точная функция')
    plt.plot(x_plot, y_L_plot, 'r--', linewidth=1.5, label=f'Лагранж L_{n-1}(x)')
    plt.plot(x_plot, y_N_plot, 'b:', linewidth=1.5, label=f'Ньютон P_{n-1}(x)')
    plt.plot(xi, yi, 'ko', markersize=8, label='узлы интерполяции')
    plt.plot([xstar], [exact], 'g*', markersize=16, label=f'x* = {xstar:.4g}')

    y_vals = list(yi) + [exact]
    ylo, yhi = min(y_vals), max(y_vals)
    ypad = 0.3 * (yhi - ylo) if yhi > ylo else 1.0
    plt.ylim(ylo - ypad, yhi + ypad)

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция: Лагранж и Ньютон')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('interpolation.png', dpi=120)
    plt.show()
    print("\nГрафик сохранён в interpolation.png")


if __name__ == "__main__":
    main()