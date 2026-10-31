# ЛР3 «Метод множителей Лагранжа», вариант 6.
# Найти глобальный минимум f(x, y, z) = x^2 + 2y^2 + 3z^2 при x + y + z = 6
# и исследовать существование максимума.
#
# Задача (f и ограничения) записана прямо в коде, как разрешено в задании. Остальное программа
# делает сама по шагам метода: строит функцию Лагранжа, берёт частные производные, решает
# систему, проверяет кандидатов подстановкой в ограничения и классифицирует их.
# Готовые оптимизаторы (scipy.optimize и т. п.) не используются.

import sys
import textwrap
from itertools import combinations

import numpy as np
import sympy as sp

EPS = 1e-8   # допуск: ограничение выполнено, если |h_j(x)| <= EPS
WIDTH = 92   # ширина строк вывода, длинные строки переносятся

# Шаги 1 и 3 алгоритма: переменные, целевая функция, ограничения в виде h_j(x) = 0
x, y, z = sp.symbols("x y z")
variables = [x, y, z]
f = x**2 + 2*y**2 + 3*z**2
constraints = [x + y + z - 6]


def say(text, indent="  "):
    """Печать с переносом длинной строки: продолжение сдвинуто ещё на два пробела."""
    print(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent + "  "))


def show(expr):
    """Выражение в привычной записи: x^2 вместо x**2."""
    return sp.sstr(expr).replace("**", "^")


def number(value):
    """Точное значение, а для дробей и корней ещё и десятичное: 36/11 ≈ 3.272727."""
    return show(value) if value.is_Integer else f"{show(value)} ≈ {float(value):.6f}"


def point_text(point, variables):
    return "(" + ", ".join(show(point[v]) for v in variables) + ")"


def feasible_example(constraints, variables):
    """Какая-нибудь допустимая точка, чтобы убедиться, что D непусто: решаем ограничения
    относительно части переменных, а остальные (свободные) берём равными 0."""
    for sol in sp.solve(constraints, variables, dict=True):
        point = {v: sp.sympify(sol.get(v, v)).subs({w: 0 for w in variables}) for v in variables}
        if all(c.is_real for c in point.values()) and \
                all(sp.simplify(h.subs(point)) == 0 for h in constraints):
            return point
    return None


def regularity(constraints, variables):
    """Матрица Якоби ограничений (строка j - градиент h_j) и особые точки: допустимые точки,
    где ранг J меньше m. Ранг меньше m, когда все миноры порядка m равны нулю."""
    J = sp.Matrix(constraints).jacobian(variables)
    m = len(constraints)
    minors = [J.extract(list(range(m)), list(cols)).det()
              for cols in combinations(range(len(variables)), m)]
    special = sp.solve(constraints + minors, variables, dict=True)
    return J, special


def lagrange_function(f, constraints):
    """Шаги 2 и 4: множители λ1..λm (по одному на ограничение) и L = f + λ1*h1 + ... + λm*hm."""
    multipliers = list(sp.symbols(f"λ1:{len(constraints) + 1}"))
    L = f + sum(lam * h for lam, h in zip(multipliers, constraints))
    return multipliers, L


def lagrange_system(L, variables, constraints):
    """Шаги 5 и 6: dL/dx_i = 0 по всем исходным переменным и сами ограничения h_j = 0
    (dL/dλ_j = h_j, поэтому производные по множителям отдельно не берём)."""
    return [sp.diff(L, v) for v in variables] + constraints


def solve_system(equations, unknowns):
    """Шаги 7 и 8: все решения системы. Решения делим на действительные, комплексные
    (их отбрасываем) и семейства с параметром (в решении осталась неизвестная).
    Если система линейная, по рангу её матрицы видно, сколько у неё решений."""
    try:
        A, b = sp.linear_eq_to_matrix(equations, unknowns)
        rank = A.rank()
        note = f"система линейная, ранг матрицы {rank}, неизвестных {len(unknowns)}"
        if rank == len(unknowns):
            note += f", определитель {A.det()} ≠ 0, поэтому решение единственное"
    except ValueError:   # sympy бросает NonlinearError (наследник ValueError)
        note = "система нелинейная"
    real, complex_, families = [], [], []
    for sol in sp.solve(equations, unknowns, dict=True):
        values = [sp.sympify(sol.get(u, u)) for u in unknowns]
        if any(v.free_symbols for v in values):
            families.append(sol)
        elif all(sp.im(v) == 0 for v in values):
            real.append(sol)
        else:
            complex_.append(sol)
    return real, complex_, families, note


def check_point(point, f, constraints):
    """Шаги 9 и 10: подстановка точки в исходные ограничения (невязки и допустимость)
    и значение целевой функции. Конечное значение f - точка в области определения."""
    residuals = [sp.simplify(h.subs(point)) for h in constraints]
    value = sp.simplify(f.subs(point))
    feasible = all(abs(complex(r)) <= EPS for r in residuals) and bool(value.is_finite)
    return residuals, feasible, value


def eigenvalues(M):
    """Собственные значения симметричной матрицы (числами)."""
    return np.linalg.eigvalsh(np.array(M, dtype=float))


def second_order(L, variables, J, solution):
    """Шаг 12, локально: гессиан L по x на касательных направлениях d (J d = 0).
    Столбцы Z - базис касательных направлений, смотрим знаки собственных значений Z^T H Z:
    все > 0 => строгий локальный минимум, все < 0 => строгий локальный максимум,
    есть разных знаков => не экстремум, иначе вторым порядком не определить."""
    H = sp.hessian(L, variables).subs(solution)
    basis = J.subs(solution).nullspace()
    if not basis:
        return None, [], "касательных направлений нет: точка изолированная"
    Z = sp.Matrix.hstack(*basis)
    R = sp.simplify(Z.T * H * Z)
    eig = eigenvalues(R)
    if eig.min() > EPS:
        verdict = "строгий локальный минимум"
    elif eig.max() < -EPS:
        verdict = "строгий локальный максимум"
    elif eig.min() < -EPS and eig.max() > EPS:
        verdict = "не экстремум (седловая точка)"
    else:
        verdict = "вторым порядком не определить"
    return R, eig, verdict


def definite_sign(H):
    """1, если матрица H не зависит от точки и положительно определена, -1 - если
    отрицательно определена, иначе 0."""
    if H.free_symbols:
        return 0
    eig = eigenvalues(H)
    return 1 if eig.min() > EPS else -1 if eig.max() < -EPS else 0


def global_extrema(f, constraints, variables, candidates):
    """Шаг 12, глобально. Ответ даём только по достаточным признакам:
    1) компактность: есть ограничение с постоянным знакоопределённым гессианом (эллипсоид,
       сфера, окружность) => D ограничено и замкнуто => по теореме Вейерштрасса минимум
       и максимум есть, их находим сравнением значений f во всех кандидатах;
    2) выпуклость: f строго выпукла (гессиан постоянный, положительно определён),
       ограничения линейные => стационарная точка - единственный глобальный минимум.
    candidates - допустимые кандидаты (точка, значение f).
    Возвращает тексты для минимума, максимума и список оснований."""
    hessians = [sp.hessian(h, variables) for h in constraints]
    H = sp.hessian(f, variables)
    values = [value for point, value in candidates]

    def where(value):
        return ", ".join(point_text(p, variables) for p, v in candidates if v == value)

    if any(definite_sign(Hh) != 0 for Hh in hessians):
        reasons = ["одно из ограничений задаёт ограниченную поверхность (её гессиан постоянный и "
                   "знакоопределённый) и D замкнуто, значит, D компактно; f непрерывна, поэтому "
                   "по теореме Вейерштрасса глобальные минимум и максимум существуют",
                   "все кандидаты найдены (регулярные и особые), поэтому глобальные экстремумы "
                   "равны наименьшему и наибольшему из значений f в кандидатах"]
        if not candidates:
            return "нет", "нет", reasons + ["но кандидатов нет: D пусто"]
        low, high = min(values), max(values)
        return (f"f = {number(low)} в точках {where(low)}",
                f"f = {number(high)} в точках {where(high)}", reasons)

    if definite_sign(H) == 1 and all(Hh.is_zero_matrix for Hh in hessians):
        dim = len(variables) - sp.Matrix(constraints).jacobian(variables).rank()
        reasons = [f"ограничения линейные, поэтому D является аффинным множеством (плоскостью) "
                   f"размерности {dim}"
                   + ("; оно неограниченное, значит, D не компактно и теорема Вейерштрасса "
                      "не применима" if dim > 0 else ""),
                   f"гессиан f = {H.tolist()} постоянный, его собственные значения "
                   f"{', '.join(f'{e:g}' for e in eigenvalues(H))} положительны, значит, "
                   f"f строго выпукла",
                   "задача выпуклая (строго выпуклая f на выпуклом D), поэтому стационарная точка "
                   "функции Лагранжа является глобальным минимумом, и он единственный"]
        if not candidates:
            return "нет", "нет", reasons + ["но кандидатов нет: D пусто"]
        point, value = candidates[0]   # у строго выпуклой задачи стационарная точка одна
        minimum = f"f{point_text(point, variables)} = {number(value)}"
        directions = sp.Matrix(constraints).jacobian(variables).nullspace()
        if not directions:
            return minimum, minimum, reasons + ["D состоит из одной точки"]
        # вдоль допустимой прямой x* + t*d функция растёт без предела => максимума нет
        d = directions[0]
        t = sp.symbols("t")
        line = sp.expand(f.subs({v: point[v] + t * d[i] for i, v in enumerate(variables)},
                                simultaneous=True))
        if sp.limit(line, t, sp.oo) == sp.oo:
            reasons.append(f"максимума нет: на допустимой прямой x* + t*d, d = {tuple(d)}, "
                           f"f = {show(line)} неограниченно растёт с ростом t, то есть "
                           f"f не ограничена сверху на D")
            return minimum, "не существует (f не ограничена сверху на D)", reasons
        return minimum, "не установлен", reasons

    return ("не установлен", "не установлен",
            ["достаточных признаков (компактность D или выпуклость задачи) программа не нашла: "
             "глобальность нужно исследовать вручную, например исключив переменную через "
             "ограничение"])


def main():
    n, m = len(variables), len(constraints)
    print("ЛР3 «Метод множителей Лагранжа», вариант 6\n")
    print(f"Целевая функция: f({', '.join(map(str, variables))}) = {show(f)}")
    print("Ограничения: " + ", ".join(f"h{j} = {show(h)} = 0" for j, h in enumerate(constraints, 1)))

    print("\nАнализ задачи:")
    smooth = all(g.is_polynomial(*variables) for g in [f] + constraints)
    if smooth:
        say(f"область определения: f и h - многочлены, они определены и бесконечно "
            f"дифференцируемы на всём R^{n}")
    else:
        say("область определения: не все функции - многочлены, её нужно проверить вручную")
    example = feasible_example(constraints, variables)
    if example:
        say(f"допустимое множество D непусто: например, точка {point_text(example, variables)}")
    else:
        say("пример допустимой точки не найден: непустоту D показывают кандидаты ниже")
    say(f"переменных n = {n}, ограничений m = {m}")
    J, special = regularity(constraints, variables)
    for j in range(m):
        say(f"градиент h{j + 1}: {tuple(J.row(j))}")
    say(f"матрица Якоби J = {J.tolist()}, нужен ранг m = {m}")
    if special:
        say("особые допустимые точки (ранг J < m), проверяются отдельно: "
            + "; ".join(point_text(p, variables) for p in special))
    else:
        say(f"особых точек нет: ранг J = {m} во всех допустимых точках, все они регулярны")
    if smooth and example and not special:
        say("вывод: f и h гладкие, D непусто, градиенты ограничений линейно независимы во всех "
            "допустимых точках, поэтому метод Лагранжа применим и даёт необходимые условия "
            "экстремума; глобальность кандидатов проверяется отдельно")
    else:
        say("вывод: метод Лагранжа даёт необходимые условия только в регулярных точках; "
            "особые точки добавлены к кандидатам, непустоту D и глобальность нужно проверить")

    multipliers, L = lagrange_function(f, constraints)
    equations = lagrange_system(L, variables, constraints)
    print("\nФункция Лагранжа: L =", show(L))
    print("Уравнения стационарности:")
    for v, eq in zip(variables, equations):
        say(f"dL/d{v} = {show(eq)} = 0")
    for j, h in enumerate(constraints, 1):
        say(f"dL/dλ{j} = h{j} = {show(h)} = 0")

    real, complex_, families, note = solve_system(equations, variables + multipliers)
    say(f"Решение системы: {note}", indent="")
    say(f"действительных решений {len(real)}, комплексных {len(complex_)} (отброшены), "
        f"семейств с параметром {len(families)}")
    for sol in families:
        say(f"семейство решений, его надо исследовать отдельно: {sol}")

    # Кандидаты: решения системы Лагранжа и особые точки
    rows = [(sol, True) for sol in real] + [(point, False) for point in special]
    candidates = []
    for k, (sol, regular) in enumerate(rows, 1):
        point = {v: sp.sympify(sol[v]) for v in variables}
        residuals, feasible, value = check_point(point, f, constraints)
        print(f"\nКандидат {k}:" if regular else f"\nКандидат {k} (особая точка, ранг J < m):")
        say("точка: " + ", ".join(f"{v} = {number(point[v])}" for v in variables))
        if regular:
            say("множители: " + ", ".join(f"{lam} = {number(sol[lam])}" for lam in multipliers))
        else:
            say("множители: нет (в особой точке система Лагранжа может не выполняться)")
        say(f"значение функции: {number(value)}")
        say("невязки ограничений: " + ", ".join(f"h{j} = {show(r)}"
                                                for j, r in enumerate(residuals, 1)))
        say("допустимость: " + ("да" if feasible else "нет"))
        if regular:
            R, eig, verdict = second_order(L, variables, J, sol)
            if R is not None:
                say(f"второй порядок: Z^T H Z = {R.tolist()}, собственные значения "
                    f"{', '.join(f'{e:.4f}' for e in eig)}, значит, {verdict}")
            else:
                say(f"второй порядок: {verdict}")
        if feasible:
            candidates.append((point, value))

    minimum, maximum, reasons = global_extrema(f, constraints, variables, candidates)
    print("\nИтог:")
    say(f"глобальный минимум: {minimum}")
    say(f"глобальный максимум: {maximum}")
    say("основание классификации:")
    for reason in reasons:
        say("- " + reason, indent="    ")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # чтобы λ и ≈ печатались и при выводе в файл
    main()
