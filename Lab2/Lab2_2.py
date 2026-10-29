import math

SAFE = {
    'x1': 0.0, 'x2': 0.0, 'a': 0.0,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'exp': math.exp, 'log': math.log, 'log10': math.log10, 'lg': math.log10,
    'sqrt': math.sqrt, 'abs': abs, 'pi': math.pi, 'e': math.e,
}


def make_func(expr):
    code = compile(expr, '<expr>', 'eval')

    def f(x1, x2):
        SAFE['x1'] = x1
        SAFE['x2'] = x2
        return float(eval(code, {'__builtins__': {}}, SAFE))

    return f


def read_input(filename):
    params = {}
    with open(filename, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            params[key.strip()] = val.strip()

    a = float(params.get('a', 0))
    SAFE['a'] = a

    f1 = make_func(params['f1'])
    f2 = make_func(params['f2'])
    phi1 = make_func(params['phi1'])
    phi2 = make_func(params['phi2'])
    x1_0 = float(params['x1_0'])
    x2_0 = float(params['x2_0'])
    eps = float(params['eps'])
    return f1, f2, phi1, phi2, x1_0, x2_0, eps


def jacobian(f1, f2, x1, x2, h=1e-7):
    df1_dx1 = (f1(x1 + h, x2) - f1(x1 - h, x2)) / (2 * h)
    df1_dx2 = (f1(x1, x2 + h) - f1(x1, x2 - h)) / (2 * h)
    df2_dx1 = (f2(x1 + h, x2) - f2(x1 - h, x2)) / (2 * h)
    df2_dx2 = (f2(x1, x2 + h) - f2(x1, x2 - h)) / (2 * h)
    return df1_dx1, df1_dx2, df2_dx1, df2_dx2


def solve_2x2(a11, a12, a21, a22, b1, b2):
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        raise ZeroDivisionError(f"det(J) = {det:.3e} - матрица Якоби вырождена")
    d1 = (a22 * b1 - a12 * b2) / det
    d2 = (-a21 * b1 + a11 * b2) / det
    return d1, d2


# Метод простой итерации
def simple_iteration(phi1, phi2, x1_0, x2_0, eps, max_iter=5000):
    x1, x2 = x1_0, x2_0
    history = []
    for k in range(1, max_iter + 1):
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)
        diff = max(abs(x1_new - x1), abs(x2_new - x2))
        history.append((k, x1_new, x2_new, diff))
        x1, x2 = x1_new, x2_new
        if diff < eps:
            return x1, x2, k, history
    return x1, x2, max_iter, history


# Метод Ньютона
def newton_method(f1, f2, x1_0, x2_0, eps, max_iter=500):
    x1, x2 = x1_0, x2_0
    history = []
    for k in range(1, max_iter + 1):
        f1v = f1(x1, x2)
        f2v = f2(x1, x2)
        a11, a12, a21, a22 = jacobian(f1, f2, x1, x2)
        dx1, dx2 = solve_2x2(a11, a12, a21, a22, -f1v, -f2v)
        x1_new = x1 + dx1
        x2_new = x2 + dx2
        diff = max(abs(x1_new - x1), abs(x2_new - x2))
        history.append((k, x1_new, x2_new, diff))
        x1, x2 = x1_new, x2_new
        if diff < eps:
            return x1, x2, k, history
    return x1, x2, max_iter, history


# Форматирование и печать
def fmt(x, decimals=10):
    if abs(x) < 10 ** (-decimals + 1):
        return f"{0:.{decimals}f}"
    return f"{x:.{decimals}f}"


def print_table(history, x1s, x2s, title):
    print(f"\n{title}")
    print(f"{'k':>3} | {'x1':>14} | {'x2':>14} | {'||dx||':>14} | {'||x-x*||':>14}")
    print("-" * 75)
    for (k, x1, x2, diff) in history:
        err = max(abs(x1 - x1s), abs(x2 - x2s))
        print(f"{k:>3} | {fmt(x1, 10):>14} | {fmt(x2, 10):>14} "
              f"| {fmt(diff, 10):>14} | {fmt(err, 10):>14}")


def main():
    filename = "lab22"
    f1, f2, phi1, phi2, x1_0, x2_0, eps = read_input(filename)

    print("Лабораторная работа 2.2: системы нелинейных уравнений")
    print(f"\nНачальное приближение: ({x1_0}, {x2_0})")
    print(f"Точность:              eps = {eps}")
    print(f"f1(x0) = {f1(x1_0, x2_0):.6f}")
    print(f"f2(x0) = {f2(x1_0, x2_0):.6f}")

    x1_star, x2_star, _, _ = newton_method(f1, f2, x1_0, x2_0, eps=1e-14, max_iter=100)
    print(f"\nЭталонное решение (Ньютон, eps=1e-14):")
    print(f"  x1* = {x1_star:.12f},  f1(x*) = {f1(x1_star, x2_star):.3e}")
    print(f"  x2* = {x2_star:.12f},  f2(x*) = {f2(x1_star, x2_star):.3e}")

    try:
        x1_s, x2_s, iter_s, hist_s = simple_iteration(phi1, phi2, x1_0, x2_0, eps)
        print(f"\nМетод Простой Итерации: {iter_s} итераций")
        print(f"  x1 = {x1_s:.10f},  f1 = {f1(x1_s, x2_s):.3e}")
        print(f"  x2 = {x2_s:.10f},  f2 = {f2(x1_s, x2_s):.3e}")
        print_table(hist_s, x1_star, x2_star, "Таблица итераций МПИ:")
    except Exception as e:
        print(f"\nМПИ: ошибка - {e}")
        x1_s, x2_s, iter_s = x1_0, x2_0, 0

    try:
        x1_n, x2_n, iter_n, hist_n = newton_method(f1, f2, x1_0, x2_0, eps)
        print(f"\nМетод Ньютона: {iter_n} итераций")
        print(f"  x1 = {x1_n:.10f},  f1 = {f1(x1_n, x2_n):.3e}")
        print(f"  x2 = {x2_n:.10f},  f2 = {f2(x1_n, x2_n):.3e}")
        print_table(hist_n, x1_star, x2_star, "Таблица итераций Ньютона:")
    except Exception as e:
        print(f"\nНьютон: ошибка — {e}")
        x1_n, x2_n, iter_n = x1_0, x2_0, 0

    print("\nСравнение")
    print(f"МПИ:    {iter_s} итераций")
    print(f"Ньютон: {iter_n} итераций")
    if iter_n > 0 and iter_s > 0:
        if iter_n < iter_s:
            print(f"Ньютон быстрее в {iter_s / iter_n:.1f} раз")
        elif iter_s < iter_n:
            print(f"МПИ быстрее в {iter_n / iter_s:.1f} раз")
        else:
            print("Одинаковое число итераций")


if __name__ == "__main__":
    main()
