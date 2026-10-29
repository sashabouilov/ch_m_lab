import math

SAFE = {
    'x': 0.0,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'exp': math.exp, 'log': math.log, 'log10': math.log10,
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

    f = make_func(params['f'])
    phi = make_func(params['phi']) if 'phi' in params else None
    df = make_func(params['df']) if 'df' in params else None
    a = float(params['a'])
    b = float(params['b'])
    eps = float(params['eps'])

    return f, phi, df, a, b, eps


def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def derivative(f, df, x):
    return df(x) if df is not None else numerical_derivative(f, x)


def locate_root(f, a, b, eps=1e-12):
    fa, fb = f(a), f(b)
    if fa * fb < 0:
        return a, b, fa, fb

    raise ValueError(
        f"На отрезке [{a}, {b}] f(a)={fa:.4g}, f(b)={fb:.4g} — знаки не разные."
    )


# Метод простой итерации
def simple_iteration(phi, a, b, eps, max_iter=1000):
    x = 0.5 * (a + b)

    h = 1e-6
    max_phi_prime = 0.0
    samples = 21
    for i in range(samples):
        t = a + (b - a) * i / (samples - 1)
        d = (phi(t + h) - phi(t - h)) / (2 * h)
        max_phi_prime = max(max_phi_prime, abs(d))

    history = []
    for k in range(1, max_iter + 1):
        x_new = phi(x)
        diff = abs(x_new - x)
        history.append((k, x_new, diff))
        x = x_new
        if diff < eps:
            return x, k, history, max_phi_prime

    return x, max_iter, history, max_phi_prime


# Метод Ньютона (касательных)
def newton_method(f, df, a, b, eps, max_iter=1000):
    h = 1e-5

    def f2(x):
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

    cands = [a, b]
    x = None
    for c in cands:
        if f(c) * f2(c) > 0:
            x = c
            break
    if x is None:
        x = 0.5 * (a + b)

    history = []
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = derivative(f, df, x)
        if abs(dfx) < 1e-15:
            raise ZeroDivisionError(f"f'(x) = 0 в точке x = {x:.6f}")
        x_new = x - fx / dfx
        diff = abs(x_new - x)
        history.append((k, x_new, diff))
        x = x_new
        if diff < eps:
            return x, k, history

    return x, max_iter, history


# Печать таблицы итераций
def fmt(x, decimals=12):
    if abs(x) < 10 ** (-decimals + 1):
        return f"{0:.{decimals}f}"
    return f"{x:.{decimals}f}"


def print_table(history, x_star, title):
    print(f"\n{title}")
    print(f"{'k':>3} | {'x_k':>18} | {'|x_k - x_{k-1}|':>18} | {'|x_k - x*|':>18}")
    print("-" * 70)
    for k, x_k, diff in history:
        err = abs(x_k - x_star)
        print(f"{k:>3} | {fmt(x_k):>18} | {fmt(diff):>18} | {fmt(err):>18}")


def main():
    filename = "lab21"
    f, phi, df, a, b, eps = read_input(filename)

    print("Лабораторная работа 2.1: нелинейные уравнения")
    print(f"\nОтрезок локализации:  a = {a},  b = {b}")
    print(f"Точность:             eps = {eps}")
    print(f"f(a) = {f(a):.6f},   f(b) = {f(b):.6f}")

    a, b, fa, fb = locate_root(f, a, b)
    print(f"Смена знака: f(a)*f(b) = {fa * fb:.3e} < 0 - корень существует")

    x_star, _, _ = newton_method(f, df, a, b, eps=1e-14, max_iter=200)
    print(f"\nЭталонное решение x* = {x_star:.12f},  f(x*) = {f(x_star):.3e}")

    x_n, iter_n, hist_n = newton_method(f, df, a, b, eps)
    print(f"\nМетод Ньютона: {iter_n} итераций, x = {x_n:.10f}, f(x) = {f(x_n):.3e}")
    print_table(hist_n, x_star, "Таблица итераций метода Ньютона:")

    if phi is None:
        raise ValueError(
            "В файле не задана функция phi(x). "
            "Приведите уравнение к виду x = phi(x) и добавьте строку phi = ..."
        )

    x_s, iter_s, hist_s, max_phi = simple_iteration(phi, a, b, eps)
    print(f"\nОценка max|phi'(x)| на отрезке: {max_phi:.4f}")
    if max_phi >= 1.0:
        print("ВНИМАНИЕ: max|phi'| >= 1 — сходимость метода не гарантирована!")
    print(f"Метод Простой Итерации: {iter_s} итераций, x = {x_s:.10f}, f(x) = {f(x_s):.3e}")
    print_table(hist_s, x_star, "Таблица итераций метода простой итерации:")

    print("\nСравнение")
    print(f"МПИ:     {iter_s} итераций,  |x - x*| = {abs(x_s - x_star):.3e}")
    print(f"Ньютон:  {iter_n} итераций,  |x - x*| = {abs(x_n - x_star):.3e}")
    if iter_n < iter_s:
        print(f"Ньютон быстрее в {iter_s / iter_n:.1f} раз")
    elif iter_s < iter_n:
        print(f"МПИ быстрее в {iter_n / iter_s:.1f} раз")
    else:
        print("Одинаковое число итераций")


if __name__ == "__main__":
    main()