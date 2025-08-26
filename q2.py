# =============================================================================
# Universidade Federal do Rio de Janeiro (UFRJ) - COPPE
# Disciplina: Mecânica de Sistemas Inteligentes (COM783)
# Aluno: Matheus Schueler de Carvalho
# Data: 25/08/2025
# Questão 2
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Métodos - questão 1
#    - rk4_step: um passo do RK clássico de 4ª ordem (passo fixo)
#    - dopri45_step: um passo embutido Dormand–Prince 5(4) (base do adaptativo)
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
        Instante atual.
    y : ndarray
        Vetor de estado no instante t.
    h : float
        Tamanho do passo de integração.

    Retorna
    -------
    y_next : ndarray
        Aproximação do estado em t+h (erro local O(h^5)).
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def dopri45_step(f, t, y, h):
    """
    Executa UM passo do esquema embutido Dormand–Prince 5(4).

    Retorna duas aproximações: ordem 5 (y_high) e ordem 4 (y_low).
    A diferença fornece uma estimativa do erro local.

    Parâmetros
    ----------
    f, t, y, h : como em rk4_step.

    Retorna
    -------
    y_high : ndarray
        Estimativa de ordem 5.
    y_low : ndarray
        Estimativa de ordem 4 (para cálculo do erro embutido).
    """
    # Coeficientes do tableau de Dormand–Prince 5(4)
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a73, a74, a75, a76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84  # a72 = 0

    # Pesos (ordem 5)
    b1, b3, b4, b5, b6 = a71, a73, a74, a75, a76
    # Pesos (ordem 4 - estimador)
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

    y_high = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)                  # ordem 5
    y_low  = y + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)  # ordem 4
    return y_high, y_low


def integrate(method, f, t_span, y0, **kwargs):
    """
    Integra o sistema y' = f(t,y), y(t0)=y0 em [t0,tf].

    Parâmetros
    ----------
    method : {'rk4','dopri45'}
        Método de integração.
    f : callable
        Campo vetorial.
    t_span : (t0, tf)
        Intervalo de integração.
    y0 : array_like
        Condição inicial.
    kwargs : parâmetros específicos de cada método.

    Retorna
    -------
    t : ndarray
        Vetor de tempos.
    y : ndarray
        Estados correspondentes.
    """
    method = method.lower()
    t0, tf = t_span
    y0 = np.asarray(y0, dtype=float)

    # ----------------- RK4 passo fixo -----------------
    if method == 'rk4':
        dt = kwargs.get('dt')
        if dt is None:
            raise ValueError("É necessário fornecer 'dt' para o RK4.")

        n = int(np.ceil((tf - t0) / dt))
        t = t0 + np.arange(n + 1) * dt
        y = np.zeros((len(y0), n + 1))
        y[:, 0] = y0
        yi = y0.copy(); ti = t0
        for i in range(n):
            yi = rk4_step(f, ti, yi, dt)
            ti += dt
            y[:, i+1] = yi
        return t, y

    # ----------------- DOPRI45 adaptativo -----------------
    elif method == 'dopri45':
        rtol    = kwargs.get('rtol',     1e-6)
        atol    = kwargs.get('atol',     1e-9)
        h_init  = kwargs.get('h_initial',1e-2)
        h_min   = kwargs.get('h_min',    1e-10)
        h_max   = kwargs.get('h_max',    np.inf)
        safety  = kwargs.get('safety',   0.9)
        max_steps = kwargs.get('max_steps', 100000000)

        direction = np.sign(tf - t0) if tf != t0 else 1.0
        h   = direction * h_init
        hmn = direction * h_min
        hmx = direction * h_max

        t = t0
        y = y0.copy()
        ts = [t]; ys = [y.copy()]
        order = 5

        for _ in range(max_steps):
            if direction * (t + h - tf) > 0:
                h = tf - t
            y_high, y_low = dopri45_step(f, t, y, h)
            e = y_high - y_low
            scale = atol + np.maximum(np.abs(y), np.abs(y_high)) * rtol
            err_norm = np.sqrt(np.sum((e / scale) ** 2) / e.size)

            if err_norm <= 1.0:
                t += h
                y = y_high
                ts.append(t); ys.append(y.copy())
                if np.allclose(t, tf):
                    break

            if err_norm == 0:
                h_new = h * 2.0
            else:
                h_new = h * safety * err_norm ** (-1.0 / (order + 1))

            factor = np.clip(h_new / h, 0.2, 5.0)
            h = np.clip(h * factor, hmn, hmx)
        else:
            raise RuntimeError("Número máximo de passos excedido.")

        return np.array(ts), np.stack(ys, axis=1)

    else:
        raise ValueError("Método desconhecido. Use 'rk4' ou 'dopri45'.")


# =============================================================================
# 2) Definição dos sistemas dinâmicos
# =============================================================================

def oscillator_rhs(zeta, wn, gamma, Omega):
    def f(t, y):
        x, v = y
        return np.array([v,
                         gamma*np.sin(Omega*t) - 2*zeta*wn*v - wn**2*x])
    return f

def lin_undamped_rhs(wn, gamma, Omega):
    def f(t, y):
        x, v = y
        return np.array([v, gamma*np.sin(Omega*t) - wn**2 * x])
    return f

def duffing_bistable_rhs(zeta, alpha, beta, gamma, Omega):
    def f(t, y):
        x, v = y
        return np.array([v, -2*zeta*v + alpha*x - beta*x**3 + gamma*np.sin(Omega*t)])
    return f

def pendulum_rhs(zeta, wn, gamma, Omega):
    def f(t, y):
        phi, w = y
        return np.array([w, -zeta*w - wn**2*np.sin(phi) + gamma*np.sin(Omega*t)])
    return f


# =============================================================================
# 3) Mapas de Poincaré
# =============================================================================

def poincare_rk4(f, y0, T, n_skip, n_pts, m_per_T=200):
    """
    Constrói o mapa de Poincaré usando RK4 de passo fixo, amostrando a solução
    no final de cada período T, após descartar um transiente.

    Parâmetros
    ----------
    f : callable
        Campo vetorial f(t, y) → dydt.
    y0 : ndarray
        Condição inicial (dimensão m, tipicamente m=2).
    T : float
        Período de excitação/observação (seção de Poincaré em t = kT).
    n_skip : int
        Número de períodos descartados (transiente).
    n_pts : int
        Número de pontos de Poincaré a registrar após o transiente.
    m_per_T : int, padrão=200
        Número de subpassos RK4 por período (dt = T/m_per_T).

    Retorna
    -------
    pts : ndarray shape (n_pts, m)
        Estados y(kT) para k = n_skip, ..., n_skip+n_pts-1.
    """
    dt = T / m_per_T
    t = 0.0
    y = y0.copy()
    pts = []
    steps_to_sample = m_per_T

    # Integra continuamente; a cada "steps_to_sample" passos, avançou-se um período.
    for k in range((n_skip + n_pts) * steps_to_sample):
        # Passo RK4 clássico
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2*k1)
        k3 = f(t + dt/2, y + dt/2*k2)
        k4 = f(t + dt,   y + dt*k3)
        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        t += dt

        # Ao completar um período, registra y(t) se já passou o transiente
        if (k + 1) % steps_to_sample == 0:
            if k >= n_skip * steps_to_sample - 1:
                pts.append(y.copy())

    return np.array(pts)


def poincare_dopri45(f, y0, T, n_skip, n_pts, **dopri_kwargs):
    """
    Mapa de Poincaré com DOPRI45 integrando *um período por vez*.
    Para evitar deriva de fase típica de métodos adaptativos, o RHS
    é avaliado com fase travada: tau = t mod T (definido localmente).
    """
    def g(t, y):
        tau = t - T * np.floor(t / T)  # tau ∈ [0,T)
        return f(tau, y)

    # escala inicial de passo, se não vier especificada
    if 'h_initial' not in dopri_kwargs:
        dopri_kwargs['h_initial'] = T / 250.0

    y = y0.copy()
    pts = []

    for i in range(n_skip + n_pts):
        # integra sempre no intervalo local [0, T]
        t_arr, y_arr = integrate('dopri45', g, (0.0, T), y, **dopri_kwargs)
        y = y_arr[:, -1]  # estado no final do período
        if i >= n_skip:
            pts.append(y.copy())

    return np.array(pts)


# =============================================================================
# 3) Plot das trajetórias no plano de fase
# =============================================================================

def trajetoria_rk4(f, y0, T, n_skip, n_pts, m_per_T=200):
    """
    Gera amostras da trajetória no plano de fase usando RK4 com o
    mesmo passo dt do mapa, apenas para fins de plote da curva contínua.

    Intervalo amostrado: [n_skip*T, (n_skip+n_pts)*T].

    Parâmetros
    ----------
    f, y0, T, n_skip, n_pts : ver poincare_rk4.
    m_per_T : int
        Número de subpassos por período (dt = T/m_per_T).

    Retorna
    -------
    xs, vs : ndarray
        Componentes da trajetória após o transiente: x(t), x˙(t).
    """
    dt = T / m_per_T
    total_steps = (n_skip + n_pts) * m_per_T
    t = 0.0
    y = y0.copy()
    xs, vs = [], []

    for k in range(total_steps):
        # Passo RK4
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2*k1)
        k3 = f(t + dt/2, y + dt/2*k2)
        k4 = f(t + dt,   y + dt*k3)
        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        t += dt

        # Armazena somente após descartar n_skip períodos
        if k >= n_skip * m_per_T:
            xs.append(y[0]); vs.append(y[1])

    return np.array(xs), np.array(vs)


def trajetoria_dopri45(f, y0, T, n_skip, n_pts, m_per_T=200, **dopri_kwargs):
    """
    Trajetória densa no plano de fase com DOPRI45 integrando período a período
    em [0,T] e reamostrando uniformemente; usa fase travada localmente
    (tau = t mod T) para que curvas de períodos repetidos se sobreponham.
    """
    def g(t, y):
        tau = t - T * np.floor(t / T)  # tau ∈ [0,T)
        return f(tau, y)

    if 'h_initial' not in dopri_kwargs:
        dopri_kwargs['h_initial'] = T / 250.0

    y = y0.copy()
    xs, vs = [], []

    for i in range(n_skip + n_pts):
        # integra um período local [0,T]
        t_arr, y_arr = integrate('dopri45', g, (0.0, T), y, **dopri_kwargs)

        # reamostra uniformemente para desenhar a linha contínua
        tt = np.linspace(0.0, T, m_per_T, endpoint=False)[1:]
        for tt_i in tt:
            j = np.searchsorted(t_arr, tt_i) - 1
            j = np.clip(j, 0, len(t_arr) - 2)
            tau = (tt_i - t_arr[j]) / (t_arr[j+1] - t_arr[j])
            y_i = y_arr[:, j] * (1 - tau) + y_arr[:, j+1] * tau
            if i >= n_skip:
                xs.append(y_i[0]); vs.append(y_i[1])

        # estado final do período vira a CI do próximo período
        y = y_arr[:, -1]

    return np.array(xs), np.array(vs)

# =============================================================================
# 4) Mapas de Poincaré para os sistemas solicitados
# =============================================================================

# Condições iniciais para cada sistema
y0_lin = np.array([0.0, 0.0])
y0_duf = np.array([0.1, 0.0])
y0_pen = np.array([0.1, 0.0])

# 1) Oscilador linear não dissipativo — testes de Ω/ω_n
wn = 1.0
gamma = 0.3
ratios = [0.5, 1.0, 1.5]
for r in ratios:
    Omega = r * wn
    T = 2*np.pi/Omega
    f = lin_undamped_rhs(wn, gamma, Omega)

    # Mapa de Poincaré (RK4) + trajetória densa para curva no plano de fase
    P = poincare_rk4(f, y0_lin, T, n_skip=300, n_pts=600, m_per_T=200)
    xs, vs = trajetoria_rk4(f, y0_lin, T, n_skip=300, n_pts=600, m_per_T=200)

    plt.figure()
    plt.plot(xs, vs, '-', linewidth=1, alpha=0.7)
    plt.plot(P[:,0], P[:,1], '.', markersize=4, label=f'Poincaré  Ω/ω_n={r:.2f}')
    plt.xlabel('x'); plt.ylabel('x˙')
    plt.title('Mapa de Poincaré — oscilador linear (RK4)')
    plt.legend(); plt.grid(True); plt.show()

# 2) Duffing biestável — varredura de γ
params_duf = [
    dict(zeta=0.05, alpha=1.0, beta=1.0, gamma=0.2,  Omega=1.0),
    dict(zeta=0.1,  alpha=1.0, beta=1.0, gamma=0.37, Omega=1.2),
    dict(zeta=0.05, alpha=1.0, beta=1.0, gamma=0.35, Omega=1.0),
]
for par in params_duf:
    T = 2*np.pi/par['Omega']
    f = duffing_bistable_rhs(**par)

    P = poincare_rk4(f, y0_duf, T, n_skip=600, n_pts=1500, m_per_T=250)
    xs, vs = trajetoria_rk4(f, y0_duf, T, n_skip=600, n_pts=1500, m_per_T=250)

    plt.figure()
    plt.plot(xs, vs, '-', linewidth=1, alpha=0.7)
    plt.plot(P[:,0], P[:,1], '.', markersize=4,
             label=f"Poincaré  γ={par['gamma']:.2f}, Ω={par['Omega']:.2f}")
    plt.xlabel('x'); plt.ylabel('x˙')
    plt.title('Mapa de Poincaré — Duffing biestável (RK4)')
    plt.legend(); plt.grid(True); plt.show()

# 3) Pêndulo forçado — varredura de (γ, Ω)
params_pen = [
    dict(zeta=0.05, wn=1.0, gamma=0.5, Omega=1.0),
    dict(zeta=0.05, wn=1.0, gamma=1.2, Omega=0.8),
    dict(zeta=0.1, wn=1.0, gamma=1.5, Omega=2.0),
]
for par in params_pen:
    T = 2*np.pi/par['Omega']
    f = pendulum_rhs(**par)

    P = poincare_rk4(f, y0_pen, T, n_skip=800, n_pts=1000, m_per_T=300)
    xs, vs = trajetoria_rk4(f, y0_pen, T, n_skip=800, n_pts=1000, m_per_T=300)

    # Para visualização do pêndulo, ângulo em (-π, π]
    phi      = (P[:,0]  + np.pi) % (2*np.pi) - np.pi
    phi_line = (xs      + np.pi) % (2*np.pi) - np.pi

    plt.figure()
    plt.plot(phi_line, vs, '-', linewidth=1, alpha=0.7)
    plt.plot(phi, P[:,1], '.', markersize=4,
             label=f"Poincaré  γ={par['gamma']:.2f}, Ω={par['Omega']:.2f}")
    plt.xlabel('ϕ (mod 2π)'); plt.ylabel('ϕ˙')
    plt.title('Mapa de Poincaré — pêndulo forçado (RK4)')
    plt.legend(); plt.grid(True); plt.show()

# 4) Duffing — RK4 × DOPRI45
par = dict(zeta=0.05, alpha=1.0, beta=1.0, gamma=0.35, Omega=1.0)
T = 2*np.pi/par['Omega']; f = duffing_bistable_rhs(**par)

P_rk = poincare_rk4(f, y0_duf, T, n_skip=800, n_pts=1200, m_per_T=250)
P_dp = poincare_dopri45(f, y0_duf, T, n_skip=800, n_pts=1200,
                        rtol=1e-9, atol=1e-11, h_initial=T/300)

xs_rk, vs_rk = trajetoria_rk4(f, y0_duf, T, n_skip=600, n_pts=1200, m_per_T=250)
xs_dp, vs_dp = trajetoria_dopri45(f, y0_duf, T, n_skip=800, n_pts=1200,
                                  m_per_T=250, rtol=1e-9, atol=1e-11, h_initial=T/300)

plt.figure()
plt.plot(xs_rk, vs_rk, '-', linewidth=1, alpha=0.6, label='trajetória RK4')
plt.plot(xs_dp, vs_dp, '-', linewidth=1, alpha=0.6, label='trajetória DOPRI45')
plt.plot(P_rk[:,0], P_rk[:,1], '.', markersize=4, label='Poincaré RK4')
plt.plot(P_dp[:,0], P_dp[:,1], '.', markersize=4, label='Poincaré DOPRI45')
plt.xlabel('x'); plt.ylabel('x˙'); plt.title('Duffing — Poincaré (RK4 × DOPRI45)')
plt.legend(); plt.grid(True); plt.show()

# 5) Pêndulo — RK4 × DOPRI45
parp = dict(zeta=0.05, wn=1.0, gamma=1.2, Omega=0.8)
T = 2*np.pi/parp['Omega']; f = pendulum_rhs(**parp)

P_rk = poincare_rk4(f, y0_pen, T, n_skip=800, n_pts=1000, m_per_T=300)
P_dp = poincare_dopri45(f, y0_pen, T, n_skip=800, n_pts=1000,
                        rtol=1e-7, atol=1e-10, h_initial=T/300)

xs_rk, vs_rk = trajetoria_rk4(f, y0_pen, T, n_skip=800, n_pts=1000, m_per_T=300)
xs_dp, vs_dp = trajetoria_dopri45(f, y0_pen, T, n_skip=800, n_pts=1000,
                                  m_per_T=300, rtol=1e-7, atol=1e-10, h_initial=T/300)

phi_rk      = (P_rk[:,0] + np.pi) % (2*np.pi) - np.pi
phi_dp      = (P_dp[:,0] + np.pi) % (2*np.pi) - np.pi
phi_line_rk = (xs_rk     + np.pi) % (2*np.pi) - np.pi
phi_line_dp = (xs_dp     + np.pi) % (2*np.pi) - np.pi

plt.figure()
plt.plot(phi_line_rk, vs_rk, '-', linewidth=1, alpha=0.6, label='trajetória RK4')
plt.plot(phi_line_dp, vs_dp, '-', linewidth=1, alpha=0.6, label='trajetória DOPRI45')
plt.plot(phi_rk, P_rk[:,1], '.', markersize=4, label='Poincaré RK4')
plt.plot(phi_dp, P_dp[:,1], '.', markersize=4, label='Poincaré DOPRI45')
plt.xlabel('ϕ (mod 2π)'); plt.ylabel('ϕ˙')
plt.title('Pêndulo — Poincaré (RK4 × DOPRI45)')
plt.legend(); plt.grid(True); plt.show()

# 6) Oscilador Linear — RK4 × DOPRI45
wn = 1.0
gamma = 0.3
Omega = 1.5
T = 2*np.pi / Omega
f = lin_undamped_rhs(wn, gamma, Omega)

# Mapa de Poincaré (RK4 vs DOPRI45) e trajetórias no plano de fase
P_rk = poincare_rk4(f, y0_lin, T, n_skip=300, n_pts=800, m_per_T=200)
P_dp = poincare_dopri45(f, y0_lin, T, n_skip=300, n_pts=800,
                        rtol=1e-9, atol=1e-11, h_initial=T/250)

xs_rk, vs_rk = trajetoria_rk4(f, y0_lin, T, n_skip=300, n_pts=800, m_per_T=200)
xs_dp, vs_dp = trajetoria_dopri45(f, y0_lin, T, n_skip=300, n_pts=800,
                                  m_per_T=200, rtol=1e-9, atol=1e-11, h_initial=T/250)

plt.figure()
plt.plot(xs_rk, vs_rk, '-', linewidth=1, alpha=0.6, label='trajetória RK4')
plt.plot(xs_dp, vs_dp, '-', linewidth=1, alpha=0.6, label='trajetória DOPRI45')
plt.plot(P_rk[:,0], P_rk[:,1], '.', markersize=4, label='Poincaré RK4')
plt.plot(P_dp[:,0], P_dp[:,1], '.', markersize=4, label='Poincaré DOPRI45')
plt.xlabel('x'); plt.ylabel('x˙')
plt.title('Oscilador linear — Poincaré (RK4 × DOPRI45)')
plt.legend(); plt.grid(True); plt.show()