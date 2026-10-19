def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n

    for i in range(1, n+1):
        row = list(map(float, lines[i].split()))
        if i == 1:
            b[0] = row[0]
            c[0] = row[1]
        elif i == n:
            a[n-1] = row[0]
            b[n-1] = row[1]
        else:
            a[i-1] = row[0]
            b[i-1] = row[1]
            c[i-1] = row[2]

    d = [float(x) for x in lines[n+1].split()]
    if len(d) != n:
        raise ValueError(
            f"Длина правой части ({len(d)}) не совпадает с n = {n}"
        )

    return n, a, b, c, d


def pq_cof(n, a, b, c, d):
    P = [0.0] * (n + 1)
    Q = [0.0] * (n + 1)

    for i in range(n):
        denom = b[i] + a[i] * P[i]
        if abs(denom) < 1e-15:
            raise ValueError(f"Деление на ноль в прогонке на шаге i={i}")
        P[i+1] = -c[i] / denom
        Q[i+1] = (d[i] - a[i] * Q[i]) / denom

    return P, Q


def back_substitution(P, Q, n):
    x = [0.0] * n
    x[n-1] = Q[n]
    for i in range(n-2, -1, -1):
        x[i] = P[i+1] * x[i+1] + Q[i+1]
    return x

def main():
    file_name = "matrix_2"
    n, a, b, c, d = read_input(file_name)
    P, Q = pq_cof(n, a, b, c, d)

    print("Прогоночные коэффициенты P:")
    for i in range(n + 1):
        print(f"  P[{i}] = {P[i]:.6f}")

    print("\nПрогоночные коэффициенты Q:")
    for i in range(n + 1):
        print(f"  Q[{i}] = {Q[i]:.6f}")

    x = back_substitution(P, Q, n)

    print("\nИтоговое решение системы:")
    for i in range(n):
        print(f"  x[{i + 1}] = {x[i]:.6f}")


if __name__ == "__main__":
    main()
