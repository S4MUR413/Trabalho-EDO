import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Função que define a EDO
def f(t, y):
    return t * np.exp(3 * t) - 2 * y

# Solução exata da EDO
def exact_solution(t):
    return (t / 5) * np.exp(3 * t) - (np.exp(-2 * t) / 25)

# Método de Euler
def euler_method(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + f(t[i-1], y[i-1]) * (t[i] - t[i-1])
    return y

# Método de Runge-Kutta de segunda ordem (aperfeiçoado)
def runge_kutta_2(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h, y[i-1] + h * k1)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

# Método de Runge-Kutta do ponto médio
def runge_kutta_midpoint(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h / 2, y[i-1] + (h / 2) * k1)
        y[i] = y[i-1] + h * k2
    return y

# Método de Runge-Kutta com parâmetros alfa, beta, p, q
def runge_kutta_custom(f, y0, t, alpha, beta, p, q):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + p * h, y[i-1] + q * h * k1)
        y[i] = y[i-1] + (alpha * k1 + beta * k2) * h
    return y

# Parâmetros para o método customizado
alpha = 0.5
beta = 0.5
p = 1
q = 1

# Configurações iniciais
y0 = 0
t = np.linspace(0, 1, 11)  # 10 intervalos, 11 pontos

# Soluções numéricas
y_euler = euler_method(f, y0, t)
y_rk2 = runge_kutta_2(f, y0, t)
y_midpoint = runge_kutta_midpoint(f, y0, t)
y_custom = runge_kutta_custom(f, y0, t, alpha, beta, p, q)
y_exact = exact_solution(t)

# Cálculo dos erros
error_euler = np.abs(y_exact - y_euler)
error_rk2 = np.abs(y_exact - y_rk2)
error_midpoint = np.abs(y_exact - y_midpoint)
error_custom = np.abs(y_exact - y_custom)

# Tabela de valores
data = {
    't': t,
    'y_exact': y_exact,
    'y_euler': y_euler,
    'y_rk2': y_rk2,
    'y_midpoint': y_midpoint,
    'y_custom': y_custom,
    'error_euler': error_euler,
    'error_rk2': error_rk2,
    'error_midpoint': error_midpoint,
    'error_custom': error_custom
}

df = pd.DataFrame(data)
print(df)

# Gráficos das soluções
plt.figure(figsize=(10, 8))
plt.plot(t, y_exact, label='Solução Exata', color='black')
plt.plot(t, y_euler, label='Euler', linestyle='--')
plt.plot(t, y_rk2, label='Runge-Kutta 2', linestyle='-.')
plt.plot(t, y_midpoint, label='Runge-Kutta Ponto Médio', linestyle=':')
plt.plot(t, y_custom, label='Runge-Kutta Customizado', linestyle='-.')

plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Soluções Aproximadas vs Solução Exata')
plt.grid(True)
plt.show()

# Gráficos dos erros
plt.figure(figsize=(10, 8))
plt.plot(t, error_euler, label='Erro Euler', linestyle='--')
plt.plot(t, error_rk2, label='Erro Runge-Kutta 2', linestyle='-.')
plt.plot(t, error_midpoint, label='Erro Runge-Kutta Ponto Médio', linestyle=':')
plt.plot(t, error_custom, label='Erro Runge-Kutta Customizado', linestyle='-.')

plt.xlabel('t')
plt.ylabel('Erro')
plt.legend()
plt.title('Erros das Soluções Aproximadas')
plt.grid(True)
plt.show()
