# Пример ввода на варианте 3:
#   max или min: max
#   Коэффициенты Z через пробел: 4 1 2 3
#   Число ограничений: 3
#   Ограничение 1: 1 1 1 0 <= 9
#   Ограничение 2: 0 1 2 1 = 7
#   Ограничение 3: 1 0 0 1 >= 3
#   Знаки переменных (>=0, <=0 или any): >=0 >=0 >=0 >=0
#
# Порядок решения:
#   1) задача приводится к каноническому виду: W => min, только равенства, b >= 0, все x >= 0;
#   2) решается вспомогательная задача с искусственными переменными;
#   3) если W'min = 0, W выражается через свободные переменные и решается основная задача.
#
# Симплекс-таблица T - список строк. Строки - базисные переменные, последняя строка это
# коэффициенты функции при свободных переменных. Последний столбец это b, в углу -Q.
# Переменные хранятся номерами: 0 это x1, 1 - x2 и т.д.

from fractions import Fraction


def read_problem():
    sense = input("max или min: ").strip()
    c = [Fraction(v) for v in input("Коэффициенты Z через пробел: ").split()]
    m = int(input("Число ограничений: "))
    A, signs, b = [], [], []
    for i in range(m):
        s = input(f"Ограничение {i + 1} (коэффициенты, знак <=, >= или =, b): ").split()
        A.append([Fraction(v) for v in s[:-2]])
        signs.append(s[-2])
        b.append(Fraction(s[-1]))
    var_signs = input("Знаки переменных (>=0, <=0 или any): ").split()
    return sense, c, A, signs, b, var_signs


def to_canonical(sense, c, A, signs, b, var_signs):
    m, n = len(A), len(c)
    c = [-v for v in c] if sense == "max" else c[:]   # max Z = min W, где W = -Z

    for i in range(m):
        if signs[i] != "=":
            k = 1 if signs[i] == "<=" else -1
            for r in range(m):
                A[r].append(Fraction(k if r == i else 0))
            c.append(Fraction(0))

    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]

    # все переменные должны быть >= 0:
    # x <= 0 заменяем на -x', x любого знака - на x' - x'' (для x'' новый столбец)
    # back[j] - пары (знак, номер новой переменной), из которых собирается исходная x_j
    back = []
    for j in range(n):
        if var_signs[j] == "<=0":
            for row in A:
                row[j] = -row[j]
            c[j] = -c[j]
            back.append([(-1, j)])
        elif var_signs[j] == "any":
            for row in A:
                row.append(-row[j])
            c.append(-c[j])
            back.append([(1, j), (-1, len(c) - 1)])
        else:
            back.append([(1, j)])
    return A, b, c, back


def step(T, r, s):
    a = T[r][s]
    new = []
    for i in range(len(T)):
        row = []
        for j in range(len(T[0])):
            if i == r and j == s:
                row.append(1 / a)
            elif i == r:
                row.append(T[r][j] / a)
            elif j == s:
                row.append(-T[i][s] / a)
            else:
                row.append(T[i][j] - T[r][j] * T[i][s] / a)   # правило прямоугольника
        new.append(row)
    return new


def simplex(T, basis, free, art=()):
    while True:
        p = T[-1]
        s = None   # разрешающий столбец: наименьший отрицательный в нижней строке
        for j in range(len(free)):
            if free[j] not in art and p[j] < 0 and (s is None or p[j] < p[s]):
                s = j
        if s is None:
            return T   # отрицательных нет

        r = None   # разрешающая строка: наименьшее b / a при a > 0
        for i in range(len(basis)):
            if T[i][s] > 0 and (r is None or T[i][-1] / T[i][s] < T[r][-1] / T[r][s]):
                r = i
        if r is None:
            return None   # в столбце нет положительных, функция не ограничена

        T = step(T, r, s)
        basis[r], free[s] = free[s], basis[r]


def solve(A, b, c):
    m, n = len(A), len(c)
    art = list(range(n, n + m))
    basis = art[:]
    free = list(range(n))
    T = [A[i] + [b[i]] for i in range(m)]
    T.append([-sum(T[i][j] for i in range(m)) for j in range(n + 1)])   # суммы столбцов с минусом
    T = simplex(T, basis, free, art)
    if T[-1][-1] < 0:
        return None, "W'min > 0, допустимых решений нет"

    # искусственная переменная могла остаться в базисе со значением 0 - выводим её
    for i in reversed(range(len(basis))):
        if basis[i] in art:
            cols = [j for j in range(len(free)) if free[j] not in art and T[i][j] != 0]
            if cols:
                T = step(T, i, cols[0])
                basis[i], free[cols[0]] = free[cols[0]], basis[i]
            else:
                del T[i], basis[i]   # строка из нулей - ограничение лишнее

    # переход к основной задаче: столбцы искусственных переменных вычёркиваем
    keep = [j for j in range(len(free)) if free[j] not in art]
    T = [[row[j] for j in keep] + [row[-1]] for row in T]
    free = [free[j] for j in keep]

    # W через свободные переменные: p_j = c_j - сумма c_баз * a_ij, в углу -Q
    cb = [c[x] for x in basis]
    rows = range(len(basis))
    T[-1] = [c[x] - sum(cb[i] * T[i][j] for i in rows) for j, x in enumerate(free)]
    T[-1].append(-sum(cb[i] * T[i][-1] for i in rows))

    T = simplex(T, basis, free)
    if T is None:
        return None, "функция не ограничена"
    return {x: T[i][-1] for i, x in enumerate(basis)}, None   # базисные = b, свободные = 0


def main():
    sense, c, A, signs, b, var_signs = read_problem()
    A, b, c2, back = to_canonical(sense, c, A, signs, b, var_signs)
    values, error = solve(A, b, c2)
    if error:
        print("Решений нет:", error)
        return
    x = [sum(k * values.get(v, 0) for k, v in pairs) for pairs in back]
    z = sum(c[j] * x[j] for j in range(len(c)))
    print("x* = (" + "; ".join(str(v) for v in x) + ")")
    print("Z" + sense, "=", z)


if __name__ == "__main__":
    main()
