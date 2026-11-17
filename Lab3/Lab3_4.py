import numpy as np

np.set_printoptions(precision=8, suppress=True)


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

    x = np.array([float(v) for v in params['x'].split(',')])
    y = np.array([float(v) for v in params['y'].split(',')])
    xstar = float(params['xstar'])
    return x, y, xstar


def find_node_index(x, xstar, tol=1e-9):
    for i in range(len(x)):
        if abs(x[i] - xstar) < tol:
            return i
    return int(np.argmin(np.abs(x - xstar)))


def select_stencil(n, i):
    if i == 0:
        return [0, 1, 2]
    if i == n - 1:
        return [n - 3, n - 2, n - 1]
    return [i - 1, i, i + 1]


def divided_differences(xn, yn):
    f0 = yn[0]
    f01 = (yn[1] - yn[0]) / (xn[1] - xn[0])
    f12 = (yn[2] - yn[1]) / (xn[2] - xn[1])
    f012 = (f12 - f01) / (xn[2] - xn[0])
    return f0, f01, f012


def compute_derivatives(xn, yn, xstar):
    f0, f01, f012 = divided_differences(xn, yn)
    df = f01 + f012 * (2 * xstar - xn[0] - xn[1])
    d2f = 2.0 * f012
    return df, d2f


def check_uniform(x, tol=1e-8):
    if len(x) < 2:
        return False, 0.0
    h = x[1] - x[0]
    for i in range(1, len(x) - 1):
        if abs((x[i + 1] - x[i]) - h) > tol:
            return False, h
    return True, h


def book_formulas_equal_h(xn, yn, i_star, idx):
    h = xn[1] - xn[0]
    if i_star == idx[0]:    # x* = a
        df = (-3 * yn[0] + 4 * yn[1] - yn[2]) / (2 * h)
    elif i_star == idx[2]:  # x* = a + 2h
        df = (3 * yn[2] - 4 * yn[1] + yn[0]) / (2 * h)
    else:                   # x* = a + h
        df = (yn[2] - yn[0]) / (2 * h)
    d2f = (yn[0] - 2 * yn[1] + yn[2]) / (h * h)
    return df, d2f


def main():
    filename = "lab34"
    x, y, xstar = read_input(filename)
    n = len(x)

    print("Лабораторная работа 3.4: Численное дифференцирование")

    print(f"\n1) Исходная таблица ({n} узлов):")
    print(f"{'i':>3} | {'x_i':>15} | {'y_i':>15}")
    for i in range(n):
        print(f"{i:>3} | {x[i]:>15.8g} | {y[i]:>15.10g}")

    print(f"\n2) Точка вычисления: X* = {xstar}")

    uniform, h = check_uniform(x)
    if uniform:
        print(f"   Сетка равномерная, шаг h = {h:.6g}")
    else:
        print(f"   Сетка неравномерная")

    i_star = find_node_index(x, xstar)
    idx = select_stencil(n, i_star)
    xn = x[idx]
    yn = y[idx]

    print(f"\n3) Использованные узлы (индексы {idx}):")
    for k, ii in enumerate(idx):
        marker = "  ← X*" if ii == i_star else ""
        print(f"   x[{ii}] = {x[ii]:.8g},  y[{ii}] = {y[ii]:.10g}{marker}")

    df, d2f = compute_derivatives(xn, yn, xstar)

    print(f"\n4) Значение первой производной:")
    print(f"   f'(X*) ≈ {df:.10g}")

    print(f"\n5) Значение второй производной:")
    print(f"   f''(X*) ≈ {d2f:.10g}")

    if uniform:
        print("\nПроверка по формулам книги для равномерной сетки\n")
        df_book, d2f_book = book_formulas_equal_h(xn, yn, i_star, idx)
        if i_star == idx[0]:
            note = "формулы (2.44a), (2.45a) — узел x0"
        elif i_star == idx[2]:
            note = "формулы (2.44c), (2.45b) — узел x2"
        else:
            note = "формулы (2.44b), (2.45b) — узел x1"
        print(f"   {note}")
        print(f"   f'(X*)  = {df_book:.10g}")
        print(f"   f''(X*) = {d2f_book:.10g}")
        ok1 = abs(df - df_book) < 1e-9
        ok2 = abs(d2f - d2f_book) < 1e-9
        print(f"   Совпадение с разделёнными разностями: "
              f"f' {'OK' if ok1 else 'FAIL'}, f'' {'OK' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()
