import ast
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

K = 1.5           # запас для L (K > 1): по сетке наклон немного занижен, поэтому умножаем на K
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
    """Оценка константы Липшица"""
    xs = np.linspace(a, b, GRID + 1)
    ys = np.array([W(x) for x in xs])
    if not np.all(np.isfinite(ys)):
        raise ValueError("функция не определена на всём отрезке (log, sqrt или деление на 0)")
    slope = np.max(np.abs(np.diff(ys)) / np.diff(xs))
    return k * slope if slope > 0 else 1.0   # у постоянной функции наклон 0, подойдёт любое L > 0


def lower_vertex(uq, Wq, uv, Wv, L):
    """Нижняя вершина между соседними верхними вершинами (uq, Wq) и (uv, Wv), uq < uv.
    Это пересечение правой ветви g_q:  W(uq) - L(x - uq)
                   и левой ветви g_v:   W(uv) + L(x - uv).
    Приравниваем их и получаем: x = (uq + uv)/2 + (W(uq) - W(uv))/(2L)."""
    x = (uq + uv) / 2 + (Wq - Wv) / (2 * L)
    p = (Wq + Wv) / 2 - L * (uv - uq) / 2      # значение ломаной в этой вершине
    return x, p


def lower_vertices(us, Ws, L):
    """Все нижние вершины заново: по одной между каждыми соседними верхними вершинами."""
    return [lower_vertex(us[j], Ws[j], us[j + 1], Ws[j + 1], L) for j in range(len(us) - 1)]


def piyavsky(W, a, b, L, eps):
    """Метод ломаных. Возвращает номер лучшей точки, все посчитанные точки в порядке
    вычисления (u_n, W(u_n), p_(n-1)(u_n)), итоговую L (которая могла вырасти), статус завершения
    и примерный минимум итоговой ломаной: настоящий минимум W лежит между ним и найденным W*."""
    us, Ws = [a, b], [W(a), W(b)]                    # верхние вершины по возрастанию u
    lows = lower_vertices(us, Ws, L)                 # lows[i] лежит между us[i] и us[i+1]
    points = [(a, Ws[0], None), (b, Ws[1], None)]    # первые две итерации: концы отрезка
    status = "max_iter"
    for n in range(2, MAX_ITER + 1):
        # нижняя вершина с наименьшим значением ломаной
        i = min(range(len(lows)), key=lambda j: lows[j][1])
        u, p = lows[i]
        # проверки новой точки: внутри своего участка, не посчитана раньше, W(u) - число
        if not us[i] <= u <= us[i + 1]:
            raise ValueError(f"нижняя вершина {u} вне участка [{us[i]}, {us[i + 1]}]")
        if u == us[i] or u == us[i + 1]:              # повтор: участок сжался до точности чисел
            status = "repeat"
            break
        Wu = W(u)
        if not np.isfinite(Wu):
            raise ValueError(f"W({u}) = {Wu}: функция не определена в этой точке")
        points.append((u, Wu, p))
        # новая верхняя вершина встаёт между us[i] и us[i+1]
        us.insert(i + 1, u)
        Ws.insert(i + 1, Wu)
        # если хорда от новой точки до соседней круче L, ломаная может пересечь график
        slope = max(abs(Wu - Ws[i]) / (u - us[i]), abs(Ws[i + 2] - Wu) / (us[i + 2] - u))
        if slope > L:
            L = K * slope                     # L занижена: увеличиваем с тем же запасом K
            lows = lower_vertices(us, Ws, L)  # старые вершины построены по старой L
            continue                          # p тоже по старой L, по ней останавливаться нельзя
        if abs(Wu - p) < eps:                 # ломаная прижалась к W в своей низшей точке
            status = "eps"
            break
        # галочка новой точки заменяет нижнюю вершину i двумя новыми
        lows[i:i + 1] = [lower_vertex(us[i], Ws[i], u, Wu, L),
                         lower_vertex(u, Wu, us[i + 2], Ws[i + 2], L)]
    best = min(range(len(points)), key=lambda k: points[k][1])   # лучшая точка за все итерации
    p_min = min(p for x, p in lows)
    return best, points, L, status, p_min


def polyline(points, L):
    """Вершины ломаной по порядку слева направо: верхняя, нижняя, верхняя, ..."""
    pts = sorted((u, Wu) for u, Wu, p in points)
    xs, ys = [pts[0][0]], [pts[0][1]]
    for (uq, Wq), (uv, Wv) in zip(pts, pts[1:]):
        x, p = lower_vertex(uq, Wq, uv, Wv, L)
        xs += [x, uv]
        ys += [p, Wv]
    return xs, ys


def print_report(text, a, b, eps, L0, L, best, points, status, p_min, t_L, t_method):
    """Вывод результата и таблицы итераций в консоль. L0 - оценка L до метода, L - после."""
    n = len(points) - 1
    print(f"\nW(x) = {text.split('=')[-1].strip()},  [a, b] = [{a:g}, {b:g}],  eps = {eps:g}")
    print(f"Константа Липшица: L = {L0:.4f} (наибольший наклон {L0 / K:.4f}, запас K = {K})")
    if L != L0:
        print(f"Хорда между посчитанными точками оказалась круче L, L увеличена до {L:.4f}")
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
    u, Wu = points[best][:2]
    if status == "eps":
        print(f"\nОстанов по точности: |W(u_{n}) - p_{n - 1}(u_{n})| = "
              f"{abs(points[-1][1] - points[-1][2]):.6f} < eps")
    elif status == "max_iter":
        print(f"\nОстанов по пределу итераций MAX_ITER = {MAX_ITER}: точность eps не достигнута")
    else:
        print("\nОстанов: следующая точка совпала с уже посчитанной, точнее посчитать нельзя, "
              "точность eps не достигнута")
    print(f"Ответ: u* = {u:.6f}, W* = {Wu:.6f} (наименьшее W среди посчитанных точек, u_{best})")
    if status != "eps":
        print(f"Настоящий минимум не ниже минимума ломаной {p_min:.6f}, "
              f"ошибка по W не больше {Wu - p_min:.6f}")
    print(f"Итераций: {n} (точки u_0 = a, u_1 = b, ..., u_{n}), вычислений W в методе: {n + 1}")
    print(f"Время: оценка L {t_L * 1000:.2f} мс, метод ломаных {t_method * 1000:.2f} мс")


def plot_result(W, text, a, b, eps, L, best, points, status, t_method, file=None):
    """Сверху: W, итоговая ломаная (при останове по точности - p_(n-1), на которой она
    достигнута), посчитанные точки и найденный минимум. Снизу: как убывала delta."""
    n = len(points) - 1
    u, Wu = points[best][:2]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[3, 1])
    xs = np.linspace(a, b, 2000)                     # густые точки только для рисования W
    ys = [W(x) for x in xs]
    top.plot(xs, ys, color="tab:blue", lw=2, label="W(x)")
    m = n - 1 if status == "eps" else n              # порядок итоговой ломаной
    lx, ly = polyline(points[:m + 1], L)
    top.plot(lx, ly, color="tab:green", lw=1, label=f"итоговая ломаная p_{m}(x)")
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
        L0 = estimate_lipschitz(W, a, b)
        t_L = time.perf_counter() - start
        start = time.perf_counter()
        best, points, L, status, p_min = piyavsky(W, a, b, L0, eps)
        t_method = time.perf_counter() - start
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as err:
        print("Ошибка:", err)
        return
    print_report(text, a, b, eps, L0, L, best, points, status, p_min, t_L, t_method)
    plot_result(W, text, a, b, eps, L, best, points, status, t_method, "result.png")
    plot_iterations(W, text, a, b, L, points, file="iterations.png")
    print("Графики сохранены в result.png и iterations.png")
    plt.show()


if __name__ == "__main__":
    main()
