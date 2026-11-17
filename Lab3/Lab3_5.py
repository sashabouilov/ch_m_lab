import math

SAFE = {
    'x': 0.0,
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
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                params[k.strip()] = v.strip()

    f = make_func(params['func'])
    a = float(params['x0'])
    b = float(params['x1'])
    h1 = float(params['h1'])
    h2 = float(params['h2'])
    exact = float(params['exact']) if 'exact' in params else None
    return f, a, b, h1, h2, exact


def grid(a, b, h):
    N_f = (b - a) / h
    N = int(round(N_f))
    if abs(N - N_f) > 1e-9:
        raise ValueError(f"h={h} не делит [{a}, {b}] нацело (N = {N_f})")
    return [a + i * h for i in range(N + 1)], N


# Пять методов интегрирования
def left_rect(f, a, b, h):
    xs, N = grid(a, b, h)
    return h * sum(f(xs[i]) for i in range(N))


def right_rect(f, a, b, h):
    xs, N = grid(a, b, h)
    return h * sum(f(xs[i]) for i in range(1, N + 1))


def mid_rect(f, a, b, h):
    xs, N = grid(a, b, h)
    return h * sum(f((xs[i] + xs[i + 1]) / 2) for i in range(N))


def trapezoid(f, a, b, h):
    xs, N = grid(a, b, h)
    s = 0.5 * (f(xs[0]) + f(xs[N]))
    for i in range(1, N):
        s += f(xs[i])
    return h * s


def simpson(f, a, b, h):
    xs, N = grid(a, b, h)
    if N % 2 != 0:
        raise ValueError(f"Симпсону нужно чётное N, получено {N}")
    s = f(xs[0]) + f(xs[N])
    for i in range(1, N):
        s += (4 if i % 2 == 1 else 2) * f(xs[i])
    return h * s / 3.0


# Метод Рунге–Ромберга
def runge_romberg(F_h2, F_h1, p, r):
    div = r ** p - 1.0
    if abs(div) < 1e-15:
        return F_h2, 0.0
    correction = (F_h2 - F_h1) / div
    refined = F_h2 + correction
    return refined, abs(correction)


def main():
    filename = "lab35"
    f, a, b, h1, h2, exact = read_input(filename)

    print("Лабораторная работа 3.5: Численное интегрирование")
    print(f"Отрезок интегрирования: [{a}, {b}]")
    print(f"Шаги: h1 = {h1}, h2 = {h2}")

    ratio = h1 / h2
    r = round(ratio)
    if abs(ratio - r) > 1e-9:
        print(f"Внимание: h1/h2 = {ratio:.6g} не целое, используются общие формулы")
        r = ratio
    else:
        print(f"Отношение шагов: r = h1/h2 = {int(r)}")

    methods = [
        ("Левые прямоуг.",   left_rect,  1),
        ("Правые прямоуг.",  right_rect, 1),
        ("Средние прямоуг.", mid_rect,   2),
        ("Трапеции",         trapezoid,  2),
        ("Симпсон",          simpson,    4),
    ]

    print("\nТаблица результатов")
    print(f"{'Метод':<20} | {'F(h1)':>18} | {'F(h2)':>18} | {'p':>2} "
          f"| {'Уточнённое (RR)':>18} | {'|R(h2)|':>14}")

    results = {}
    for name, fn, p in methods:
        try:
            F1 = fn(f, a, b, h1)
            F2 = fn(f, a, b, h2)
            refined, err2 = runge_romberg(F2, F1, p, r)
            results[name] = (F1, F2, p, refined, err2)
            print(f"{name:<20} | {F1:>18.10g} | {F2:>18.10g} | {p:>2} "
                  f"| {refined:>18.10g} | {err2:>14.4e}")
        except Exception as e:
            print(f"{name:<20} | ОШИБКА: {e}")

    if exact is not None:
        print("\nСраснение с точным значением")
        print(f"Точное значение интеграла: F = {exact:.12g}\n")
        print(f"{'Метод':<20} | {'|F(h2) - F|':>14} | {'|RR - F|':>14} "
              f"| {'Отн. погр. F(h2)':>18}")

        best_name = None
        best_err = float('inf')
        for name, (F1, F2, p, refined, err2) in results.items():
            err_h2 = abs(F2 - exact)
            err_rr = abs(refined - exact)
            rel = err_h2 / abs(exact) if abs(exact) > 1e-15 else float('nan')
            print(f"{name:<20} | {err_h2:>14.4e} | {err_rr:>14.4e} "
                  f"| {rel:>18.4e}")
            if err_h2 < best_err:
                best_err = err_h2
                best_name = name

        print(f"\nНаиболее точный метод (по F(h2)): {best_name}")
        print(f"Его абсолютная погрешность:       {best_err:.4e}")


if __name__ == "__main__":
    main()