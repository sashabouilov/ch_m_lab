import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    eps = float(lines[1])
    A = np.array([list(map(float, lines[i + 2].split())) for i in range(n)])
    return n, eps, A


def householder_qr(A):
    n = A.shape[0]
    R = A.copy().astype(float)
    Q = np.eye(n)

    for k in range(n - 1):
        v = R[k:, k].copy()
        norm_v = np.sqrt(np.sum(v * v))
        if norm_v < 1e-15:
            continue
        if v[0] >= 0:
            v[0] += norm_v
        else:
            v[0] -= norm_v
        v_norm = np.sqrt(np.sum(v * v))
        if v_norm < 1e-15:
            continue
        v = v / v_norm

        H = np.eye(n)
        H[k:, k:] -= 2.0 * np.outer(v, v)

        R = H @ R
        Q = Q @ H

    return Q, R


def extract_eigenvalues(A, block_eps):
    n = A.shape[0]
    eigvals = []
    i = 0
    while i < n:
        if i == n - 1:
            eigvals.append(complex(A[i, i]))
            i += 1
        elif abs(A[i + 1, i]) < block_eps:
            eigvals.append(complex(A[i, i]))
            i += 1
        else:
            a, b = A[i, i], A[i, i + 1]
            c, d = A[i + 1, i], A[i + 1, i + 1]
            trace = a + d
            det = a * d - b * c
            disc = trace * trace - 4 * det
            if disc >= 0:
                sq = np.sqrt(disc)
                eigvals.append(complex((trace + sq) / 2))
                eigvals.append(complex((trace - sq) / 2))
            else:
                sq = np.sqrt(-disc)
                eigvals.append(complex(trace / 2, sq / 2))
                eigvals.append(complex(trace / 2, -sq / 2))
            i += 2
    return eigvals


def qr_algorithm(A, eps, max_iter=5000):
    n = A.shape[0]
    A_k = A.copy().astype(float)

    block_eps = max(eps, 1e-6)
    prev_eigs = None
    it = 0

    for it in range(1, max_iter + 1):
        cur_eigs = extract_eigenvalues(A_k, block_eps)
        if prev_eigs is not None:
            cur_sorted = sorted(cur_eigs,
                                key=lambda z: (round(z.real, 6), round(z.imag, 6)))
            prev_sorted = sorted(prev_eigs,
                                 key=lambda z: (round(z.real, 6), round(z.imag, 6)))
            max_diff = max(abs(cur_sorted[j] - prev_sorted[j]) for j in range(n))
            if max_diff < eps:
                break
        prev_eigs = cur_eigs

        Q, R = householder_qr(A_k)
        A_k = R @ Q

    return A_k, it


def fmt(z, eps=1e-7):
    re, im = z.real, z.imag
    if abs(im) < eps:
        return f"{re:+.6f}"
    sign = "+" if im >= 0 else "-"
    return f"{re:+.6f} {sign} {abs(im):.6f}i"


def print_eigvals(eigs, header):
    print(header)
    for l in eigs:
        print(f"  {fmt(l)}")


def main():
    filename = "matrix_5"
    n, eps, A = read_input(filename)

    print("=" * 60)
    print(f"Размерность: n = {n}")
    print(f"Точность: eps = {eps}")
    print("\nИсходная матрица A:")
    print(A)

    print("\n" + "=" * 60)
    print("QR-разложение")
    print("=" * 60)

    Q, R = householder_qr(A)
    print("Матрица Q:")
    print(Q)
    print("\nМатрица R:")
    print(R)

    QR = Q @ R
    print("\nQ · R:")
    print(QR)
    print(f"\nПроверка Q·R = A: {np.allclose(QR, A)}")
    print(f"Проверка Q^T·Q = E: {np.allclose(Q.T @ Q, np.eye(n), atol=1e-10)}")

    print("\n" + "=" * 60)
    print("QR-алгоритм для собственных значений")
    print("=" * 60)

    A_final, iters = qr_algorithm(A, eps)
    print(f"Число итераций: {iters}")
    print("\nМатрица после итераций:")
    print(A_final)

    eigenvalues = extract_eigenvalues(A_final, max(eps, 1e-6))
    print_eigvals(eigenvalues, "\nНайденные собственные значения:")

    print("\n" + "=" * 60)
    print("Проверка через NumPy")
    print("=" * 60)

    eig_ref = np.linalg.eigvals(A)
    print_eigvals(eig_ref, "Эталонные собственные значения:")

    print("\n" + "=" * 60)
    print("Комплексно-сопряжённые пары")
    print("=" * 60)

    eig_list = list(eigenvalues)
    used = [False] * len(eig_list)
    found = False
    for i in range(len(eig_list)):
        if used[i]:
            continue
        l = eig_list[i]
        if abs(l.imag) < 1e-6:
            continue
        for j in range(i + 1, len(eig_list)):
            if used[j]:
                continue
            m = eig_list[j]
            if abs(l.real - m.real) < 1e-4 and abs(l.imag + m.imag) < 1e-4:
                print(f"  Найдена комплексно-сопряжённая пара:")
                print(f"    λ₁ = {fmt(l)}")
                print(f"    λ₂ = {fmt(m)}")
                used[i] = True
                used[j] = True
                found = True
                break
    if not found:
        print("  Комплексно-сопряжённых пар не найдено.")


if __name__ == "__main__":
    main()
