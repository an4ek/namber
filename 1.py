import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Исходная функция из задания:
# (0.2x)^3 - cos(x) = 0
def f(x):
    return (0.2 * x) ** 3 - np.cos(x)

# Первая производная
# f'(x) = 3*(0.2x)^2 * 0.2 + sin(x) = 0.024x^2 + sin(x)
def df(x):
    return 0.024 * x**2 + np.sin(x)

# Вторая производная
# f''(x) = 0.048x + cos(x)
def ddf(x):
    return 0.048 * x + np.cos(x)


# 1. графическое отделение корней
x = np.linspace(-6, 6, 400)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = (0.2x)^3 - cos(x)')
plt.axhline(0, color='black', linewidth=0.7)
plt.axvline(0, color='black', linewidth=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = (0.2x)^3 - cos(x)')
plt.grid(True)
plt.legend()
plt.show()


# графики производных
x_vals = np.linspace(-6, 6, 400)
f_vals = f(x_vals)
df_vals = df(x_vals)
ddf_vals = ddf(x_vals)

plt.figure(figsize=(10, 6))

# Первая производная
plt.subplot(2, 1, 1)
plt.plot(x_vals, df_vals, label="f'(x)", color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Первая производная f'(x)")
plt.legend()
plt.grid()

# Вторая производная
plt.subplot(2, 1, 2)
plt.plot(x_vals, ddf_vals, label="f''(x)", color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.title("Вторая производная f''(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# функция для простой итерации (через формулу Ньютона)
def phi(x):
    return x - f(x) / df(x)


# Метод дихотомии
def bisection(a, b, eps, max_iter=100):
    iteration = 0
    while (b - a) / 2 > eps and iteration < max_iter:#остановка
        c = (a + b) / 2
        if f(a) * f(c) > 0:
            a = c
        else:
            b = c
        iteration += 1
    return (a + b) / 2, iteration

# Метод простой итерации
def simple_iteration(x0, eps, max_iter=100):
    x_prev = x0
    x_curr = phi(x_prev)
    iteration = 0
    while abs(x_curr - x_prev) > eps and iteration < max_iter:#остановка
        x_prev = x_curr
        x_curr = phi(x_prev)
        iteration += 1
    return x_curr, iteration

#Метод Ньютона
def newton(x0, eps, max_iter=100):
    x = x0
    iteration = 0
    while abs(f(x)) > eps and iteration < max_iter:#остановка
        x = x - f(x) / df(x)
        iteration += 1
    return x, iteration

#Метод хорд
def secant(x0, x1, eps, max_iter=100):
    iteration = 0
    while abs(x1 - x0) > eps and iteration < max_iter:#остановка
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0 = x1
        x1 = x2
        iteration += 1
    return x1, iteration

#Метод Чебышева
def chebyshev(x0, eps, max_iter=100):
    x = x0
    iteration = 0
    while abs(f(x)) > eps and iteration < max_iter:#остановка
        x = x - f(x)/df(x) - 0.5 * (f(x)/df(x))**2 * (ddf(x)/df(x))
        iteration += 1
    return x, iteration


# 2. уточнение двух корней
eps = 1e-5

# Интервалы
interval1 = (-2, 0)
interval2 = (1, 3)

# Начальные приближения
x0_1 = -1.0
x0_2 = 2.0

# Дихотомия
root_bisection1, it_bis1 = bisection(interval1[0], interval1[1], eps)
root_bisection2, it_bis2 = bisection(interval2[0], interval2[1], eps)

# Простая итерация
root_iter1, it_iter1 = simple_iteration(x0_1, eps)
root_iter2, it_iter2 = simple_iteration(x0_2, eps)

# Ньютон
root_newton1, it_newton1 = newton(x0_1, eps)
root_newton2, it_newton2 = newton(x0_2, eps)

# Хорды
root_secant1, it_secant1 = secant(interval1[0], interval1[1], eps)
root_secant2, it_secant2 = secant(interval2[0], interval2[1], eps)

# Чебышев
root_cheb1, it_cheb1 = chebyshev(x0_1, eps)
root_cheb2, it_cheb2 = chebyshev(x0_2, eps)


# 3. проверка корректности (f(x) ≈ 0)
print("Корень 1:")
print(f"  Дихотомия: x = {root_bisection1:.5f}, итераций = {it_bis1}, f(x) = {f(root_bisection1):.2e}")
print(f"  Простая итерация: x = {root_iter1:.5f}, итераций = {it_iter1}, f(x) = {f(root_iter1):.2e}")
print(f"  Ньютон: x = {root_newton1:.5f}, итераций = {it_newton1}, f(x) = {f(root_newton1):.2e}")
print(f"  Хорды: x = {root_secant1:.5f}, итераций = {it_secant1}, f(x) = {f(root_secant1):.2e}")
print(f"  Чебышев: x = {root_cheb1:.5f}, итераций = {it_cheb1}, f(x) = {f(root_cheb1):.2e}")

print("\nКорень 2:")
print(f"  Дихотомия: x = {root_bisection2:.5f}, итераций = {it_bis2}, f(x) = {f(root_bisection2):.2e}")
print(f"  Простая итерация: x = {root_iter2:.5f}, итераций = {it_iter2}, f(x) = {f(root_iter2):.2e}")
print(f"  Ньютон: x = {root_newton2:.5f}, итераций = {it_newton2}, f(x) = {f(root_newton2):.2e}")
print(f"  Хорды: x = {root_secant2:.5f}, итераций = {it_secant2}, f(x) = {f(root_secant2):.2e}")
print(f"  Чебышев: x = {root_cheb2:.5f}, итераций = {it_cheb2}, f(x) = {f(root_cheb2):.2e}")