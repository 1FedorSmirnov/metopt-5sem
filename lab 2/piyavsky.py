"""
Лабораторная работа 2 «Метод ломаных» (метод Пиявского).

Программа ищет глобальный минимум липшицевой функции W(x) на отрезке [a, b] с точностью eps.

Ввод - с клавиатуры (python piyavsky.py, программа сама спросит) или четырьмя аргументами:
    python piyavsky.py "x + sin(3.14159*x)" -1.3 4.5 0.01
  функция - формула от x, например 4*x + cos(9*pi*x). Умножение только со знаком: 4*x, а не 4x.
            Степень ^ или **, можно sin, cos, tan, exp, log, sqrt, abs, pi, e. «f(x) =» не обязательно.
            В командной строке функция в кавычках, с клавиатуры - без кавычек;
  a, b    - концы отрезка, a < b;
  eps     - точность, eps > 0. Дробные числа - с точкой или с запятой: 0.01 или 0,01.

Обозначения как на лекции:
  W(x)                              - исходная функция;
  u_n                               - точки, где посчитана W (верхние вершины ломаной);
  g_n(x, u_n) = W(u_n) - L|x - u_n| - основной конструкционный элемент ломаной («галочка»);
  p_n(x) = max{p_(n-1)(x), g_n(x)}  - ломаная n-го порядка, миноранта: она нигде не выше W(x);
  нижняя вершина                    - точка пересечения двух соседних галочек.

Ход метода:
  1) оцениваем константу Липшица L - насколько круто может меняться W;
  2) итерации 0 и 1: u_0 = a, u_1 = b, ломаная p_1(x) = max{g_0, g_1};
  3) следующая точка u_n - нижняя вершина с наименьшим значением ломаной;
  4) если |W(u_n) - p_(n-1)(u_n)| < eps, то останов: u* = u_n, W* = W(u_n);
  5) иначе добавляем галочку g_n: она заменяет нижнюю вершину u_n двумя новыми, и снова п. 3.
"""
import ast
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

K = 1.5           # запас для L: оценка по сетке немного занижена, поэтому умножаем на K
GRID = 10000      # на сколько частей делим отрезок при оценке L
MAX_ITER = 5000   # предел итераций, чтобы при очень маленьком eps программа не работала вечно

# Что можно писать в функции, кроме x, чисел и + - * / ** (^ тоже считается степенью)
NAMES = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
         "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e}
NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


def make_function(text):
    """Строку вида "f(x) = x + sin(3.14159*x)" превращает в функцию W(x)."""
    text = text.split("=")[-1].replace("^", "**").strip()   # "f(x) =" можно не писать
    tree = ast.parse(text, mode="eval")
    # eval выполнит любой код, поэтому сначала проверяем, что в строке только формула
    for node in ast.walk(tree):
        if not isinstance(node, NODES) or (isinstance(node, ast.Name) and node.id not in NAMES
                                           and node.id != "x"):
            raise ValueError("в функции можно писать только x, числа, + - * / ** и "
                             + ", ".join(NAMES))
    code = compile(tree, "W", "eval")
    return lambda x: float(eval(code, {"__builtins__": {}}, {**NAMES, "x": x}))


def estimate_lipschitz(W, a, b, k=K):
    """Оценка константы Липшица: наибольший наклон W между соседними точками мелкой сетки,
    то есть численная производная max|W'(x)|, умноженная на запас k.
    Сетка нужна только для L, минимум по ней не ищется."""
    xs = np.linspace(a, b, GRID + 1)
    ys = np.array([W(x) for x in xs])
    if not np.all(np.isfinite(ys)):
        raise ValueError("функция не определена на всём отрезке (log, sqrt или деление на 0)")
    slope = np.max(np.abs(np.diff(ys)) / np.diff(xs))
    return k * slope if slope > 0 else 1.0   # у постоянной функции наклон 0, подойдёт любое L > 0


def lower_vertex(uq, Wq, uv, Wv, L):
    """Нижняя вершина между соседними верхними вершинами (uq, Wq) и (uv, Wv), uq < uv.
    Это пересечение правой ветви галочки g_q:  W(uq) - L(x - uq)
                   и левой ветви галочки g_v:   W(uv) + L(x - uv).
    Приравниваем их и получаем формулу с лекции: x = (uq + uv)/2 + (W(uq) - W(uv))/(2L)."""
    x = (uq + uv) / 2 + (Wq - Wv) / (2 * L)
    p = (Wq + Wv) / 2 - L * (uv - uq) / 2      # значение ломаной в этой вершине
    return x, p


def piyavsky(W, a, b, L, eps):
    """Метод ломаных. Возвращает u*, W*, номер последней итерации, все посчитанные точки
    в порядке вычисления (u_n, W(u_n), p_(n-1)(u_n)) и признак, что L не оказалась занижена."""
    us, Ws = [a, b], [W(a), W(b)]                    # верхние вершины по возрастанию u
    lows = [lower_vertex(a, Ws[0], b, Ws[1], L)]     # lows[i] лежит между us[i] и us[i+1]
    points = [(a, Ws[0], None), (b, Ws[1], None)]    # итерации 0 и 1: концы отрезка
    L_ok = True
    for n in range(2, MAX_ITER + 1):
        # нижняя вершина с наименьшим значением ломаной: только здесь W может быть ниже всего
        i = min(range(len(lows)), key=lambda j: lows[j][1])
        u, p = lows[i]
        Wu = W(u)
        points.append((u, Wu, p))
        if abs(Wu - p) < eps:                         # ломаная прижалась к W в своей низшей точке
            return u, Wu, n, points, L_ok
        # если между точками W меняется круче L, ломаная может пересечь график: L занижена
        if abs(Wu - Ws[i]) > L * (u - us[i]) or abs(Ws[i + 1] - Wu) > L * (us[i + 1] - u):
            L_ok = False
        # новая верхняя вершина встаёт между us[i] и us[i+1],
        # а её галочка заменяет нижнюю вершину i двумя новыми
        us.insert(i + 1, u)
        Ws.insert(i + 1, Wu)
        lows[i:i + 1] = [lower_vertex(us[i], Ws[i], u, Wu, L),
                         lower_vertex(u, Wu, us[i + 2], Ws[i + 2], L)]
    print(f"Точность eps не достигнута за {MAX_ITER} итераций, ответ - лучшая найденная точка")
    u, Wu, p = min(points, key=lambda t: t[1])
    return u, Wu, MAX_ITER, points, L_ok


def polyline(points, L):
    """Вершины ломаной по порядку слева направо: верхняя, нижняя, верхняя, ...
    Ломаная рисуется прямо по ним, сетка не нужна."""
    pts = sorted((u, Wu) for u, Wu, p in points)
    xs, ys = [pts[0][0]], [pts[0][1]]
    for (uq, Wq), (uv, Wv) in zip(pts, pts[1:]):
        x, p = lower_vertex(uq, Wq, uv, Wv, L)
        xs += [x, uv]
        ys += [p, Wv]
    return xs, ys


def print_report(text, a, b, eps, L, u, Wu, n, points, L_ok, t_L, t_method):
    """Вывод результата и таблицы итераций в консоль."""
    print(f"\nW(x) = {text.split('=')[-1].strip()},  [a, b] = [{a:g}, {b:g}],  eps = {eps:g}")
    print(f"Константа Липшица: L = {L:.4f} (наибольший наклон {L / K:.4f}, запас K = {K})")
    print("\n   n          u_n       W(u_n)  p_(n-1)(u_n)     delta")
    rows = list(enumerate(points))
    shown = rows if len(rows) <= 15 else rows[:10] + [None] + rows[-4:]
    for row in shown:
        if row is None:
            print("  ...")
            continue
        k, (uk, Wk, pk) = row
        if pk is None:
            print(f"{k:4d} {uk:12.6f} {Wk:12.6f}   (конец отрезка)")
        else:
            print(f"{k:4d} {uk:12.6f} {Wk:12.6f} {pk:12.6f} {abs(Wk - pk):10.6f}")
    print(f"\nОстанов: |W(u_{n}) - p_{n - 1}(u_{n})| = {abs(points[-1][1] - points[-1][2]):.6f} < eps")
    print(f"Ответ: u* = {u:.6f}, W* = {Wu:.6f}")
    print(f"Итераций: {n} (точки u_0 = a, u_1 = b, ..., u_{n}), вычислений W в методе: {n + 1}")
    print(f"Время: оценка L {t_L * 1000:.2f} мс, метод ломаных {t_method * 1000:.2f} мс")
    if not L_ok:
        print("Внимание: W местами круче L, константа Липшица занижена, ответ может быть неверным")


def plot_result(W, text, a, b, eps, L, u, Wu, n, points, t_method, file=None):
    """Сверху: W, итоговая ломаная p_(n-1), на которой достигнута точность, посчитанные
    точки и найденный минимум. Снизу: как от итерации к итерации убывала delta."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[3, 1])
    xs = np.linspace(a, b, 2000)                     # густые точки только для рисования W
    ys = [W(x) for x in xs]
    top.plot(xs, ys, color="tab:blue", lw=2, label="W(x)")
    lx, ly = polyline(points[:-1], L)
    top.plot(lx, ly, color="tab:green", lw=1, label=f"итоговая ломаная p_{n - 1}(x)")
    top.scatter([t[0] for t in points], [t[1] for t in points], s=12, color="black", zorder=4,
                label="посчитанные точки u_n (верхние вершины)")
    top.scatter([u], [Wu], marker="*", s=300, color="red", zorder=5,
                label=f"минимум: u* = {u:.4f}, W* = {Wu:.4f}")
    low, high = min(min(ly), min(ys)), max(ys)
    top.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    top.set_title(f"Метод ломаных: W(x) = {text.split('=')[-1].strip()} на [{a:g}, {b:g}]\n"
                  f"eps = {eps:g}, L = {L:.3f}, итераций {n}, время метода {t_method * 1000:.2f} мс")
    top.set_xlabel("x")
    top.legend(fontsize=8, framealpha=0.9)
    top.grid(alpha=0.3)

    deltas = [abs(Wk - pk) for uk, Wk, pk in points[2:]]
    bottom.semilogy(range(2, n + 1), deltas, ".-", color="tab:purple", label="delta_n = |W(u_n) - p_(n-1)(u_n)|")
    bottom.axhline(eps, color="red", ls="--", label=f"eps = {eps:g}")
    bottom.set_xlabel("итерация n")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=150)


def plot_iterations(W, text, a, b, L, points, count=6, file=None):
    """Первые итерации: ломаная p_n, вспомогательные функции g_0 ... g_n (серые галочки)
    и следующая точка u_(n+1) - нижняя вершина с наименьшим значением ломаной."""
    count = min(count, len(points) - 2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    xs = np.linspace(a, b, 2000)
    ys = [W(x) for x in xs]
    for k, ax in enumerate(axes.flat[:count], start=1):
        known = points[:k + 1]                       # u_0 ... u_k
        ax.plot(xs, ys, color="tab:blue", lw=1.5)
        for uk, Wk, pk in known:                     # галочки g_i на всём отрезке
            ax.plot([a, uk, b], [Wk - L * (uk - a), Wk, Wk - L * (b - uk)], color="gray", lw=0.6)
        lx, ly = polyline(known, L)
        ax.plot(lx, ly, color="tab:green", lw=2)
        ax.scatter([t[0] for t in known], [t[1] for t in known], s=15, color="black", zorder=4)
        un, Wn, pn = points[k + 1]                   # следующая точка
        ax.plot([un, un], [pn, Wn], "r--", lw=1)
        ax.scatter([un], [pn], color="red", zorder=5)
        ax.set_ylim(min(ly) - 0.05 * (max(ys) - min(ly)), max(ys) + 0.05 * (max(ys) - min(ly)))
        ax.set_title(f"p_{k}(x), следующая точка u_{k + 1} = {un:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle(f"Первые итерации: W(x) = {text.split('=')[-1].strip()} (синяя), "
                 f"галочки g_n (серые), ломаная p_n (зелёная)")
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=150)


def read_input():
    """Функция, концы отрезка и eps: из командной строки или с клавиатуры."""
    if len(sys.argv) == 5:
        text, a, b, eps = sys.argv[1:]
    else:
        text = input("Функция, например f(x) = x + sin(3.14159*x): ")
        a = input("Левый конец отрезка a: ")
        b = input("Правый конец отрезка b: ")
        eps = input("Точность eps, например 0,01: ")
    a, b, eps = (float(s.replace(",", ".")) for s in (a, b, eps))   # можно с запятой
    if a >= b:
        raise ValueError("левый конец должен быть меньше правого")
    if eps <= 0:
        raise ValueError("eps должно быть больше 0")
    return text, make_function(text), a, b, eps


def main():
    try:
        text, W, a, b, eps = read_input()
        start = time.perf_counter()
        L = estimate_lipschitz(W, a, b)
        t_L = time.perf_counter() - start
        start = time.perf_counter()
        u, Wu, n, points, L_ok = piyavsky(W, a, b, L, eps)
        t_method = time.perf_counter() - start
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as err:
        print("Ошибка:", err)
        return
    print_report(text, a, b, eps, L, u, Wu, n, points, L_ok, t_L, t_method)
    plot_result(W, text, a, b, eps, L, u, Wu, n, points, t_method, "result.png")
    plot_iterations(W, text, a, b, L, points, file="iterations.png")
    print("Графики сохранены в result.png и iterations.png")
    plt.show()


if __name__ == "__main__":
    main()
