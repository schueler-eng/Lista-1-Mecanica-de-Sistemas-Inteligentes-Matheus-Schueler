# =============================================================================
# Universidade Federal do Rio de Janeiro (UFRJ) - COPPE
# Disciplina: Mecânica de Sistemas Inteligentes (COM783)
# Aluno: Matheus Schueler de Carvalho
# Data: 25/08/2025
# Questão 1
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Implementação dos métodos solicitados
#    - rk4_step: implementação de um passo do RK clássico de 4ª ordem (passo fixo)
#    - dopri45_step: implementação de um passo embutido Dormand–Prince 5(4) (base do adaptativo)
#    - integrate: com a lógica de integração
# =============================================================================

def rk4_step(f, t, y, h):
    """
    Executa UM passo do método de Runge–Kutta clássico de 4ª ordem.
    Parâmetros
    ----------
    f : callable
        Campo vetorial f(t, y) → dydt.
    t : float
        Instante de avaliação.
    y : ndarray (m,)
        Estado atual.
    h : float
        Tamanho do passo.

    Retorna
    -------
    y_next : ndarray (m,)
        Aproximação de y(t+h) com erro local O(h^5).
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def dopri45_step(f, t, y, h):
    """
    Executa UM passo do esquema embutido Dormand–Prince 5(4).
    Produz duas aproximações: ordem 5 (y_high) e ordem 4 (y_low).
    A diferença y_high - y_low fornece estimativa do erro local.

    Parâmetros
    ----------
    f, t, y, h : como em rk4_step.

    Retorna
    -------
    y_high : ndarray (m,)
        Estimativa de ordem 5.
    y_low  : ndarray (m,)
        Estimativa de ordem 4 para cálculo do erro embutido.
    """
    # Coeficientes do tableau de Dormand–Prince 5(4)
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a73, a74, a75, a76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84  # a72 = 0

    # Pesos da solução de ordem 5
    b1, b3, b4, b5, b6 = a71, a73, a74, a75, a76
    # Pesos da solução de ordem 4 (para o estimador de erro)
    b1s, b3s, b4s, b5s, b6s, b7s = (
        5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
    )

    # Estágios
    k1 = f(t, y)
    k2 = f(t + c2*h, y + h * a21 * k1)
    k3 = f(t + c3*h, y + h * (a31 * k1 + a32 * k2))
    k4 = f(t + c4*h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = f(t + c5*h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = f(t + c6*h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    k7 = f(t + h,     y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

    # Soluções embutidas
    y_high = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)                  # ordem 5
    y_low  = y + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)  # ordem 4
    return y_high, y_low


# =============================================================================
# 2) Estrutura de integração (malhas, controle de erro, despacho de método)
# =============================================================================
def integrate(method, f, t_span, y0, **kwargs):
    """
    Integra o IVP y' = f(t,y), y(t0) = y0, em t ∈ [t0, tf].

    Parâmetros
    ----------
    method : {'rk4', 'dopri45'}
        'rk4'      → passo fixo com RK4 clássico (usa rk4_step).
        'dopri45'  → passo adaptativo com Dormand–Prince 5(4) embutido (usa dopri45_step).
    f : callable
        Campo vetorial f(t, y) → dydt.
    t_span : tuple (t0, tf)
        Intervalo de integração.
    y0 : array_like (m,)
        Condição inicial.
    kwargs :
        Parâmetros específicos de cada método:
        - rk4:    dt
        - dopri45: rtol, atol, h_initial, h_min, h_max, safety, max_steps

    Retorna
    -------
    t : ndarray (N,)
        Nós temporais da integração.
    y : ndarray (m, N)
        Trajetória numérica correspondente.
    """
    method = method.lower()
    t0, tf = t_span
    y0 = np.asarray(y0, dtype=float)

    # -------------------------- RK4 (passo fixo) --------------------------
    if method == 'rk4':
        dt = kwargs.get('dt')
        if dt is None:
            raise ValueError("'dt' parameter is required for rk4 integration")

        n = int(np.ceil((tf - t0) / dt))
        t = t0 + np.arange(n + 1) * dt
        y = np.zeros((len(y0), n + 1))
        y[:, 0] = y0

        yi = y0.copy()
        ti = t0
        for i in range(n):
            yi = rk4_step(f, ti, yi, dt)
            ti += dt
            y[:, i+1] = yi
        return t, y

    # --------------------- DOPRI45 (passo adaptativo) ---------------------
    elif method == 'dopri45':
        # Tolerâncias de controle de erro (norma RMS ponderada componente a componente)
        rtol      = kwargs.get('rtol',      1e-6)
        atol      = kwargs.get('atol',      1e-9)
        h_init    = kwargs.get('h_initial', 1e-2)
        h_min     = kwargs.get('h_min',     1e-10)
        h_max     = kwargs.get('h_max',     np.inf)
        safety    = kwargs.get('safety',    0.9)
        max_steps = kwargs.get('max_steps', 100000000)

        # Integração pode ser crescente (tf>t0) ou decrescente (tf<t0)
        direction = np.sign(tf - t0) if tf != t0 else 1.0
        h   = direction * h_init
        hmn = direction * h_min
        hmx = direction * h_max

        t = t0
        y = y0.copy()
        ts = [t]
        ys = [y.copy()]
        order = 5  # ordem do método "alto" (usado na lei de passo)

        for _ in range(max_steps):
            # Ajuste do último passo para coincidir exatamente com tf
            if direction * (t + h - tf) > 0:
                h = tf - t

            # Passo embutido e estimativa de erro
            y_high, y_low = dopri45_step(f, t, y, h)
            e = y_high - y_low
            # Escala por componente: erro relativo (rtol) + erro absoluto (atol)
            scale = atol + np.maximum(np.abs(y), np.abs(y_high)) * rtol
            err_norm = np.sqrt(np.sum((e / scale) ** 2) / e.size)  # norma RMS

            if err_norm <= 1.0:
                # Passo aceito: avança o tempo e armazena o estado
                t += h
                y = y_high
                ts.append(t)
                ys.append(y.copy())
                if np.allclose(t, tf):
                    break

            # Atualização do passo conforme a teoria de controle embutido p/(p+1)
            if err_norm == 0:
                h_new = h * 2.0
            else:
                h_new = h * safety * err_norm ** (-1.0 / (order + 1))

            factor = h_new / h
            factor = np.clip(factor, 0.2, 5.0)  # limitações padrão para evitar oscilações
            h = np.clip(h * factor, hmn, hmx)
        else:
            raise RuntimeError("Maximum number of steps exceeded")

        return np.array(ts), np.stack(ys, axis=1)

    else:
        raise ValueError(f"Unknown integration method '{method}'. Use 'rk4' or 'dopri45'.")


# =============================================================================
# 3) Problema-teste: oscilador harmônico linear amortecido e forçado
#    Funções auxiliares: RHS em 1ª ordem e solução analítica para validação.
# =============================================================================
def oscillator_rhs(zeta, wn, gamma, Omega):
    """
    Retorna f(t,y) para o sistema em 1ª ordem com y = [x, v]^T.
    """
    def f(t, y):
        x, v = y
        return np.array([
            v,
            gamma * np.sin(Omega * t) - 2 * zeta * wn * v - wn**2 * x
        ])
    return f


def analytic_solution(t, zeta, wn, gamma, Omega, x0, v0):
    """
    Solução analítica para ζ<1: soma de transitório subamortecido e regime forçado.
    Usada para computar o erro RMS dos métodos.
    """
    wd  = wn * np.sqrt(1 - zeta**2)
    X   = gamma / np.sqrt((wn**2 - Omega**2)**2 + (2 * zeta * wn * Omega)**2)
    phi = np.arctan2(2 * zeta * wn * Omega, wn**2 - Omega**2)

    A = x0 + X * np.sin(phi)
    B = (v0 - X * Omega * np.cos(phi) + zeta * wn * A) / wd

    x = (np.exp(-zeta * wn * t) *
         (A * np.cos(wd * t) + B * np.sin(wd * t)) +
         X * np.sin(Omega * t - phi))
    return x


# =============================================================================
# 4) Experimentos numéricos e visualização
# =============================================================================

# Parâmetros físicos e condição inicial do problema-teste
zeta = 0.05
wn   = 2 * np.pi        # 1 Hz
Omega = 1.2 * wn
gamma = 1.0
x0, v0 = 0.0, 0.0
y0 = np.array([x0, v0])
t_end = 20.0
f = oscillator_rhs(zeta, wn, gamma, Omega)

# ---- Histórias temporais: RK4 (passo fixo) vs DOPRI45 (adaptativo) ----
dt = 0.01
t_rk4, y_rk4 = integrate('rk4', f, (0.0, t_end), y0, dt=dt)
t_dp,  y_dp  = integrate('dopri45', f, (0.0, t_end), y0,
                         rtol=1e-6, atol=1e-9, h_initial=0.008)

# Solução analítica para comparação
t_analytic = np.linspace(0.0, t_end, 1000)
x_analytic = analytic_solution(t_analytic, zeta, wn, gamma, Omega, x0, v0)

# Gráfico: respostas no tempo
plt.figure()
plt.plot(t_analytic, x_analytic, label='Analítico')
plt.plot(t_rk4,  y_rk4[0], '--', label=f'RK4 Δt={dt} s')
plt.plot(t_dp,   y_dp[0],  '--', label='DOPRI45 adapt.')
plt.xlabel('Tempo [s]')
plt.ylabel('Deslocamento x(t)')
plt.title('Oscilador amortecido e forçado')
plt.legend()
plt.grid(True)
plt.show()

# ---- Estudo de convergência: RK4 (vs Δt) ----
dts = [0.02, 0.01, 0.005, 0.0025, 0.001, 0.0005]
errors_rk4 = []
for d in dts:
    t_rk, y_rk = integrate('rk4', f, (0.0, t_end), y0, dt=d)
    x_rk = analytic_solution(t_rk, zeta, wn, gamma, Omega, x0, v0)
    errors_rk4.append(np.sqrt(np.mean((y_rk[0] - x_rk) ** 2)))

plt.figure()
plt.loglog(dts, errors_rk4, marker='o', label='RK4 (erro vs Δt)')
plt.xlabel('Δt [s]')
plt.ylabel('Erro RMS')
plt.title('Convergência RK4 (passo fixo)')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.show()

# ---- Estudo de convergência: DOPRI45 (vs tolerância) ----
# A análise é feita variando rtol (atol proporcional) e observando
# a redução do erro RMS e o custo (nº de passos aceitos).
rtols = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
errors_dp  = []
nsteps_dp  = []

for r in rtols:
    a = 1e-3 * r       # escolha proporcional; ajusta a sensibilidade absoluta
    t_dp_tmp, y_dp_tmp = integrate('dopri45', f, (0.0, t_end), y0,
                                   rtol=r, atol=a,
                                   h_initial=0.008, h_min=1e-10, h_max=np.inf, safety=0.9)
    x_dp = analytic_solution(t_dp_tmp, zeta, wn, gamma, Omega, x0, v0)
    errors_dp.append(np.sqrt(np.mean((y_dp_tmp[0] - x_dp) ** 2)))
    nsteps_dp.append(len(t_dp_tmp) - 1)  # custo: nº de passos aceitos

# Erro RMS em função da tolerância
plt.figure()
plt.loglog(rtols, errors_dp, marker='s', label='DOPRI45 (erro vs rtol)')
plt.xlabel('rtol')
plt.ylabel('Erro RMS')
plt.title('Convergência DOPRI45 (adaptativo) — erro vs tolerância')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.show()

# Custo computacional em função da tolerância
plt.figure()
plt.loglog(rtols, nsteps_dp, marker='^', label='Passos aceitos')
plt.xlabel('rtol')
plt.ylabel('Nº de passos')
plt.title('DOPRI45 — custo vs tolerância')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.show()

# ---- Evolução temporal do passo adaptativo (integração principal) ----
dt_dp = np.diff(t_dp)
t_mid = t_dp[:-1] + dt_dp / 2
plt.figure()
plt.plot(t_mid, dt_dp, marker='o')
plt.xlabel('Tempo [s]')
plt.ylabel('Passo adaptativo Δt')
plt.title('Adaptação do passo (DOPRI45)')
plt.grid(True)
plt.show()