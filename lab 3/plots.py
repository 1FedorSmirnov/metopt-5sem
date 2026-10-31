# ЛР3, дополнительное задание: графики к варианту 6, геометрический смысл метода Лагранжа.
# 3d: плоскость x + y + z = 6 (допустимое множество D), поверхность уровня
#   f = f* (эллипсоид), которая касается плоскости в точке минимума, линии уровня f на плоскости
#   и градиенты f и h в точке касания: они лежат на одной прямой. Справа та же картина,
#   если смотреть вдоль плоскости: плоскость видна ребром, эллипсоид касается её в одной точке.
# plane: линии уровня f на самой плоскости (z выражено через x и y) и f вдоль
#   допустимой прямой x* + t*d: функция растёт без предела, поэтому максимума нет.
# Точка, множитель и f* берутся из решения lagrange.py, а не вписаны руками.

import os

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lagrange import f, variables, constraints, lagrange_function, lagrange_system, solve_system

HERE = os.path.dirname(os.path.abspath(__file__))
# подписи с формулами в математической записи ($...$), шрифт как в отчёте
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})


def tex(expr):
    """Выражение для подписи на графике: x^{2} + 2 y^{2}, дроби в строку (72/11)."""
    return sp.latex(expr, fold_short_frac=True)

# Решение системы Лагранжа (то же, что в lagrange.py)
multipliers, L = lagrange_function(f, constraints)
real, complex_, families, note = solve_system(lagrange_system(L, variables, constraints),
                                              variables + multipliers)
sol = real[0]
h = constraints[0]
p = np.array([float(sol[v]) for v in variables])               # x* - точка минимума
lam = float(sol[multipliers[0]])                               # множитель Лагранжа
f_star = float(f.subs(sol))                                    # f* = f(x*)
grad_f = np.array([float(sp.diff(f, v).subs(sol)) for v in variables])
grad_h = np.array([float(sp.diff(h, v).subs(sol)) for v in variables])
A = np.array(sp.hessian(f, variables), dtype=float) / 2        # f = x^T A x
w, Q = np.linalg.eigh(A)
# ортонормированный базис e1, e2 плоскости: из касательных направлений J d = 0
J = sp.Matrix(constraints).jacobian(variables)
E = np.linalg.qr(np.array(sp.Matrix.hstack(*J.nullspace()), dtype=float))[0]
e1, e2 = E[:, 0], E[:, 1]
levels = f_star + np.array([2, 6, 12, 20, 30])                # уровни f для линий на плоскости
lam_text = rf"$\lambda^* = {tex(sol[multipliers[0]])} \approx {lam:.3f}$"
f_text = rf"$f^* = {tex(f.subs(sol))} \approx {f_star:.3f}$"
point_text = r"$x^* = (" + r",\ ".join(tex(sol[v]) for v in variables) + ")$"


def vector(g):
    """Градиент функции g в точке x* точными числами: (72/11, 72/11, 72/11)."""
    return "(" + r",\ ".join(tex(sp.diff(g, v).subs(sol)) for v in variables) + ")"


def level_curve(c, count=200):
    """Линия уровня f = c на плоскости: точки x* + r*u, u = cos(s)*e1 + sin(s)*e2.
    На плоскости f(x* + r*u) = f* + r^2 * u^T A u (линейной части нет, grad f перпендикулярен
    плоскости), поэтому r = sqrt((c - f*) / u^T A u)."""
    s = np.linspace(0, 2 * np.pi, count)
    u = np.outer(e1, np.cos(s)) + np.outer(e2, np.sin(s))
    r = np.sqrt((c - f_star) / np.einsum("is,ij,js->s", u, A, u))
    return p[:, None] + u * r


def ellipsoid(c, n_theta=60, n_phi=30):
    """Поверхность уровня f = c: x^T A x = c, эллипсоид с полуосями sqrt(c / w_i)."""
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, n_theta), np.linspace(0, np.pi, n_phi))
    sphere = np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
    pts = Q @ (np.sqrt(c / w)[:, None] * sphere.reshape(3, -1))
    return pts.reshape(3, *th.shape)


def plot_3d(file):
    fig = plt.figure(figsize=(10, 5.9))
    views = [(12, -25, "persp", "Общий вид"),
             (0, -45, "ortho", r"Вид вдоль плоскости, по направлению $(1,\ -1,\ 0)$:" "\n"
                               r"плоскость видна ребром, эллипсоид касается её в $x^*$")]
    for k, (elev, azim, proj, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        ax.set_proj_type(proj)
        U, V = np.meshgrid(np.linspace(-5.2, 5.2, 2), np.linspace(-5.2, 5.2, 2))
        plane = p[:, None, None] + e1[:, None, None] * U + e2[:, None, None] * V
        ax.plot_surface(*plane, color="tab:blue", alpha=0.18, edgecolor="none")
        ax.plot_surface(*ellipsoid(f_star), color="tab:orange", alpha=0.30, edgecolor="none",
                        rstride=1, cstride=1)
        for c in levels:
            ax.plot(*level_curve(c), color="tab:green", lw=1)
        ax.quiver(*p, *grad_f, length=3.2, normalize=True, color="red", lw=2.2,
                  arrow_length_ratio=0.15)
        ax.quiver(*p, *grad_h, length=1.6, normalize=True, color="blue", lw=2.2,
                  arrow_length_ratio=0.25)
        ax.scatter(*p, color="red", s=45, depthshade=False)
        ax.text(*(p + [0.3, -0.3, -0.6]), r"$x^*$", color="red", fontsize=13)
        ax.scatter(0, 0, 0, color="black", s=25, depthshade=False)
        ax.text(0.2, 0.2, -0.9, r"$(0,\ 0,\ 0)$", fontsize=10)
        ax.set_xlim(-5, 7)
        ax.set_ylim(-5, 7)
        ax.set_zlim(-5, 7)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        if proj == "ortho":            # оси x и y здесь видны сбоку, их подписи накладываются
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_title(title, fontsize=11)
    handles = [Patch(color="tab:blue", alpha=0.35, label=f"допустимое множество $D$: ${tex(h)} = 0$"),
               Patch(color="tab:orange", alpha=0.5, label=f"поверхность уровня $f = f^*$, {f_text}"),
               Line2D([], [], color="tab:green", label="линии уровня $f$ на плоскости ($f > f^*$)"),
               Line2D([], [], color="red", lw=2,
                      label=rf"$\nabla f(x^*) = {vector(f)} = -\lambda^* \nabla h(x^*)$"),
               Line2D([], [], color="blue", lw=2, label=rf"$\nabla h(x^*) = {vector(h)}$, {lam_text}"),
               Line2D([], [], color="red", marker="o", ls="none", label=point_text),
               Line2D([], [], color="black", marker="o", ls="none", markersize=4,
                      label=r"$(0,\ 0,\ 0)$: безусловный минимум $f$ лежит вне плоскости")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"$f = {tex(f)}$,  ${tex(h)} = 0$: в точке минимума $x^*$ эллипсоид $f = f^*$ "
                 "касается плоскости $D$", fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.25, wspace=0.0)
    fig.savefig(file, dpi=150, bbox_inches="tight")


def plot_plane(file):
    x, y, z = variables
    F = f.subs(z, sp.solve(h, z)[0])                    # f на плоскости: z = 6 - x - y
    Fn = sp.lambdify((x, y), F, "numpy")
    dFx, dFy = (sp.lambdify((x, y), sp.diff(F, v), "numpy") for v in (x, y))
    d = np.array(J.nullspace()[0], dtype=float).ravel()  # допустимое направление, как в lagrange.py
    t = sp.symbols("t")
    line = sp.expand(f.subs({v: sol[v] + t * int(d[i]) for i, v in enumerate(variables)},
                            simultaneous=True))
    g = sp.lambdify(t, line, "numpy")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), width_ratios=[1.15, 1])
    X, Y = np.meshgrid(np.linspace(-2.5, 9, 400), np.linspace(-3.5, 7, 400))
    cs = left.contour(X, Y, Fn(X, Y), levels=f_star + np.array([1, 4, 9, 16, 25, 36, 49, 64]),
                      cmap="viridis")
    left.clabel(cs, fmt="%.0f", fontsize=8)
    # стрелки - grad f на плоскости (касательная часть): куда f растёт вдоль D
    GX, GY = np.meshgrid(np.linspace(-1.5, 8, 9), np.linspace(-2.5, 6, 9))
    left.quiver(GX, GY, dFx(GX, GY), dFy(GX, GY), color="gray", alpha=0.6, width=0.003)
    ts = np.linspace(-4, 4, 2)
    left.plot(p[0] + ts * d[0], p[1] + ts * d[1], "--", color="tab:red", lw=1,
              label=rf"прямая $x^* + t d$, $d = ({d[0]:g},\ {d[1]:g},\ {d[2]:g})$")
    left.plot(*p[:2], "r*", markersize=16, label=r"$x^*$: здесь касательная часть $\nabla f$ равна 0")
    left.set_xlabel("$x$")
    left.set_ylabel("$y$")
    left.set_aspect("equal")
    left.set_title("Линии уровня $f$ на плоскости $D$ ($z = 6 - x - y$)\n"
                   "серые стрелки: в какую сторону $f$ растёт вдоль $D$", fontsize=11)
    left.legend(fontsize=9, loc="upper right", framealpha=0.95)
    left.grid(alpha=0.3)

    T = np.linspace(-4, 4, 400)
    right.plot(T, g(T), color="tab:purple", lw=2, label=f"$f(x^* + t d) = {tex(line)}$")
    right.plot([0], [f_star], "r*", markersize=16, label=f"$t = 0$: минимум, {f_text}")
    right.set_xlabel("$t$")
    right.set_ylabel("$f$")
    right.set_title("$f$ вдоль допустимой прямой растёт без предела,\nзначит, максимума нет",
                    fontsize=11)
    right.legend(fontsize=9, loc="upper center")
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file, dpi=150)


if __name__ == "__main__":
    plot_3d(os.path.join(HERE, "lagrange_3d.png"))
    plot_plane(os.path.join(HERE, "lagrange_plane.png"))
    print("Графики сохранены в lagrange_3d.png и lagrange_plane.png")
    plt.show()
