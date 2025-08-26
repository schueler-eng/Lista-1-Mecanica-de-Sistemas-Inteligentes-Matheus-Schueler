# =============================================================================
# Universidade Federal do Rio de Janeiro (UFRJ) - COPPE
# Disciplina: Mecânica de Sistemas Inteligentes (COM783)
# Aluno: Matheus Schueler de Carvalho
# Data: 25/08/2025
# Questão 3
# =============================================================================

"""
Q3 – Pontos de equilíbrio e estabilidade local
----------------------------------------------
Este script implementa um código para:
  • Encontrar pontos de equilíbrio de sistemas autônomos  ẏ = g(y)
    (busca por Newton multivariável com jacobiano numérico).
  • Classificar a natureza local dos equilíbrios a partir do espectro de J = ∂g/∂y.
  • Visualizar, para sistemas 2D (ou seções 2D), um 'heatmap' baseado no campo
    linearizado que destaca a influência de cada equilíbrio no entorno.

Sistemas solicitados (enunciado Q3):
  (a) Duffing autônomo        – y = [x, v]
  (b) Pêndulo simples (ζ≥0)   – y = [φ, ω]
  (c) Lorenz (3D)
  (d) Sistema multiestável 2‑GL – y = [x1, v1, x2, v2]

"""

import numpy as np
import itertools
from numpy.linalg import norm, eigvals
import math
import matplotlib.pyplot as plt

# =============================================================================
# 1) Utilidades numéricas
# =============================================================================
def numerical_jacobian(g, y, eps=1e-7):
    """
    Aproxima o jacobiano J = ∂g/∂y em y por diferenças progressivas.

    Parâmetros
    ----------
    g : callable
        Campo vetorial g(y) → ℝ^m.
    y : array_like
        Ponto onde J será avaliado.
    eps : float
        Passo relativo para a perturbação.

    Retorna
    -------
    J : ndarray (m×n)
        Jacobiano numérico de g no ponto y.
    """
    y = np.asarray(y, dtype=float)
    f0 = g(y)
    m = len(f0); n = len(y)
    J = np.zeros((m, n))
    for j in range(n):
        yj = y.copy()
        h = eps * max(1.0, abs(y[j]))
        yj[j] += h
        J[:, j] = (g(yj) - f0) / h
    return J


def newton_nd(g, y0, tol=1e-10, maxit=50):
    """
    Newton multivariável com backtracking para resolver g(y)=0.

    Parâmetros
    ----------
    g : callable
        Campo vetorial g(y).
    y0 : array_like
        Chute inicial.
    tol : float
        Tolerância em norma infinito para ||g(y)||_∞.
    maxit : int
        Máximo de iterações de Newton.

    Retorna
    -------
    y  : ndarray
        Solução (aproximada).
    ok : bool
        True se convergiu (||g(y)||_∞ < tol).
    """
    y = np.asarray(y0, dtype=float)
    for _ in range(maxit):
        F = g(y)
        if norm(F, np.inf) < tol:
            return y, True
        J = numerical_jacobian(g, y)
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Quase-singular: usa pseudo-inverso como passo de correção
            dy = -np.linalg.pinv(J) @ F
        # Backtracking de Armijo simples
        alpha = 1.0
        for _ in range(10):
            y_new = y + alpha * dy
            if norm(g(y_new), np.inf) < norm(F, np.inf):
                y = y_new
                break
            alpha *= 0.5
        else:
            # Não houve redução suficiente
            return y, False
    return y, False


def uniq_rows(arr, tol=1e-7):
    """
    Remove duplicatas de uma coleção de vetores (linhas), com tolerância.

    Parâmetros
    ----------
    arr : ndarray (k×n)
        Conjunto de candidatos a solução.
    tol : float
        Tolerância absoluta (L∞) para considerar dois pontos como iguais.

    Retorna
    -------
    out : ndarray
        Conjunto reduzido sem duplicatas numéricas.
    """
    out = []
    for r in arr:
        if not any(np.allclose(r, s, atol=tol, rtol=0) for s in out):
            out.append(r)
    return np.array(out)


def scan_equilibria(g, box, grid_per_dim=9, tol_root=1e-9):
    """
    Faz uma varredura de sementes numa 'caixa' cartesiana e roda Newton.

    Parâmetros
    ----------
    g : callable
        Campo vetorial g(y).
    box : list of (min,max)
        Intervalo por coordenada (dimensão n).
    grid_per_dim : int
        Nº de sementes por dimensão (grade uniforme).
    tol_root : float
        Tolerância do Newton.

    Retorna
    -------
    eqs : ndarray (q×n)
        Equilíbrios distintos encontrados.
    """
    seeds = [np.linspace(b[0], b[1], grid_per_dim) for b in box]
    eqs = []
    for y0 in itertools.product(*seeds):
        y, ok = newton_nd(g, np.array(y0, float), tol=tol_root, maxit=60)
        if ok:
            eqs.append(y)
    if len(eqs) == 0:
        return np.empty((0, len(box)))
    return uniq_rows(np.array(eqs))


def classify_equilibrium(g, y_eq):
    """
    Classifica o equilíbrio y* por autovalores do jacobiano J(y*).

    Parâmetros
    ----------
    g : callable
        Campo vetorial.
    y_eq : array_like
        Ponto de equilíbrio.

    Retorna
    -------
    J   : ndarray
        Jacobiano em y*.
    lam : ndarray
        Autovalores de J.
    cls : str
        Texto com a classificação ('assintoticamente estável', 'sela', etc.).
    """
    J = numerical_jacobian(g, y_eq)
    lam = eigvals(J)

    # Classificação geral (qualquer dimensão)
    if np.any(np.real(lam) > 0):
        cls = "instável (autovalor com parte real > 0)"
    elif np.all(np.real(lam) < 0):
        cls = "assintoticamente estável"
    else:
        cls = "neutra / caso limite (linearização inconclusiva)"

    # Refinamento para n=2 (diagrama traço×determinante)
    if len(y_eq) == 2:
        tr = np.trace(J)
        det = np.linalg.det(J)
        disc = tr**2 - 4*det
        if det < 0:
            cls = "sela (instável)"
        elif det > 0:
            if tr < 0:
                cls = "nó estável" if disc > 0 else ("foco (espiral) estável" if disc < 0 else "nó degenerado estável")
            elif tr > 0:
                cls = "nó instável" if disc > 0 else ("foco (espiral) instável" if disc < 0 else "nó degenerado instável")
            else:
                cls = "centro (estável no sentido de Lyapunov)"
    return J, lam, cls

# =============================================================================
# 2) Estilo de marcadores para os equilíbrios
# =============================================================================
def _kind_from_cls(cls: str) -> str:
    s = cls.lower()
    if "sela" in s: return "saddle"
    if "assintoticamente estável" in s or "nó estável" in s or "foco (espiral) estável" in s: return "stable"
    if "centro" in s or "neutra" in s or "inconclusiva" in s: return "center"
    if "instável" in s: return "unstable"
    return "unknown"

def _plot_marker(ax, x, y, kind, labels_done):
    """
    Plota o marcador e escreve a legenda apenas uma vez por 'kind'.
    """
    if kind == "stable":
        h = ax.plot(x, y, marker='o', linestyle='None', mfc='black', mec='black', ms=8,
                    label=None if labels_done.get(kind) else "equilíbrio estável")
    elif kind == "unstable":
        h = ax.plot(x, y, marker='o', linestyle='None', mfc='white', mec='red', ms=8, mew=1.8,
                    label=None if labels_done.get(kind) else "equilíbrio instável")
    elif kind == "saddle":
        h = ax.plot(x, y, marker='x', linestyle='None', color='red', ms=9, mew=2.0,
                    label=None if labels_done.get(kind) else "sela")
    elif kind == "center":
        h = ax.plot(x, y, marker='o', linestyle='None', mfc='white', mec='blue', ms=8, mew=1.8,
                    label=None if labels_done.get(kind) else "centro / neutro")
    else:
        h = ax.plot(x, y, marker='^', linestyle='None', mfc='none', mec='k', ms=8,
                    label=None if labels_done.get(kind) else "indefinido")
    labels_done[kind] = True
    return h

# =============================================================================
# 3) Sistemas dinâmicos do enunciado
# =============================================================================
# (a) Duffing (autônomo): x¨ + 2ζ x˙ + α x + β x^3 = 0  -> y=[x, v]
def duffing_aut_rhs(zeta=0.1, alpha=1.0, beta=1.0):
    """
    Retorna g(y) do Duffing autônomo com amortecimento linear.
    y = [x, v] → [v, -2ζ v - α x - β x^3].
    """
    def g(y):
        x, v = y
        return np.array([v, -2*zeta*v - alpha*x - beta*x**3])
    return g

# (b) Pêndulo simples (ζ ≥ 0): φ¨ + ζ φ˙ + ω_n^2 sin φ = 0  -> y=[φ, ω]
def pendulum_aut_rhs(wn=1.0, zeta=0.0):
    """
    Retorna g(y) do pêndulo simples com amortecimento opcional.
    • ζ=0: sistema conservativo (centros em φ*=2kπ).
    • ζ>0: dissipativo (minimos tornam-se nós/focos estáveis).
    """
    def g(y):
        phi, w = y
        return np.array([w, -zeta*w - wn**2*np.sin(phi)])
    return g

# (c) Lorenz 3D
def lorenz_rhs(sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Clássico sistema de Lorenz.
    """
    def g(y):
        x, yv, z = y
        return np.array([sigma*(yv - x), x*(rho - z) - yv, x*yv - beta*z])
    return g

# (d) 2‑GL multiestável (sem forçamento): y=[x1,v1,x2,v2]
def multi2dof_rhs(z1=0.05, z2=0.05, a1=-1.2, b1=1.0, a2=-0.8, b2=1.0, rho=1.0, Os=1.0):
    """
    Sistema acoplado de 2 graus de liberdade com não linearidades cúbicas.
    """
    def g(y):
        x1, v1, x2, v2 = y
        dv1 = -2*z1*v1 + 2*z2*(v2 - v1) - (1 + a1)*x1 - b1*x1**3 + rho*Os**2*(x2 - x1)
        dv2 = (1.0/rho)*(-2*z2*(v2 - v1) - a2*x2 - b2*x2**3 - rho*Os**2*(x2 - x1))
        return np.array([v1, dv1, v2, dv2])
    return g

# =============================================================================
# 4) Heatmaps a partir do campo linearizado
# =============================================================================
def heatmap_global_linearized_2d(ax, eqs, Js, xlim, ylim, res=220, log=True):
    """
    Calcula e plota o escalar
        S(x,y) = min_i || J_i * ([x,y]^T - y_i^*) ||,
    que mede, para cada ponto do plano, a intensidade do campo linearizado
    relativo ao equilíbrio mais 'influente'. Útil para visualizar bacias locais.
    """
    xs = np.linspace(*xlim, res); ys = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xs, ys)
    S = None
    for ystar, J in zip(eqs, Js):
        DD = np.stack([XX - ystar[0], YY - ystar[1]], axis=-1).reshape(-1, 2)  # (N,2)
        VV = (J @ DD.T).T                                                       # (N,2)
        Si = np.linalg.norm(VV, axis=1).reshape(res, res)
        S = Si if S is None else np.minimum(S, Si)
    if log:
        S = np.log10(S + 1e-14)
    im = ax.imshow(S, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\min_i \|J_i(y-y_i^\star)\|$" + (" (log10)" if log else ""))
    return im


def heatmap_global_lorenz_slice(ax, eqs, Js, xlim, ylim, res=220, log=True):
    """
    Fatia 2D no plano (x,y) com deslocamento dz=0:
        S(x,y) = min_i || J_i · [x-x_i^*, y-y_i^*, 0]^T ||.
    """
    xs = np.linspace(*xlim, res); ys = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xs, ys)
    S = None
    for ystar, J in zip(eqs, Js):
        DD = np.stack([XX - ystar[0], YY - ystar[1], np.zeros_like(XX)], axis=-1).reshape(-1,3)
        VV = (J @ DD.T).T
        Si = np.linalg.norm(VV, axis=1).reshape(res, res)
        S = Si if S is None else np.minimum(S, Si)
    if log:
        S = np.log10(S + 1e-14)
    im = ax.imshow(S, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\min_i \|J_i[dx,dy,0]^T\|$" + (" (log10)" if log else ""))
    return im


def heatmap_global_2gl(ax, eqs, Js, x1lim, x2lim, res=220, log=True):
    """
    Projeção (x1,x2) com dv1=dv2=0:
        S(x1,x2) = min_i sqrt( (δẋ̇1)^2 + (δẋ̇2)^2 )
    derivado do campo linearizado 4D.
    """
    xs = np.linspace(*x1lim, res); ys = np.linspace(*x2lim, res)
    XX, YY = np.meshgrid(xs, ys)
    S = None
    for ystar, J in zip(eqs, Js):
        DD = np.stack([XX - ystar[0], np.zeros_like(XX), YY - ystar[2], np.zeros_like(XX)], axis=-1).reshape(-1,4)
        VV = (J @ DD.T).T
        Si = np.sqrt(VV[:,1]**2 + VV[:,3]**2).reshape(res, res)
        S = Si if S is None else np.minimum(S, Si)
    if log:
        S = np.log10(S + 1e-14)
    im = ax.imshow(S, origin='lower', extent=(x1lim[0], x1lim[1], x2lim[0], x2lim[1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\min_i \sqrt{(\delta\dot v_1)^2+(\delta\dot v_2)^2}$" + (" (log10)" if log else ""))
    return im

# =============================================================================
# 5) Pipelines (uma figura por sistema, apenas heatmap + equilíbrios)
# =============================================================================
def plot_duffing_heatmap():
    """
    Varre combinações (α,β) representativas e plota o heatmap global 2D com
    os equilíbrios marcados e classificados no plano (x, x˙).
    """
    print("\n(a) Duffing — heatmap global + equilíbrios")
    for (alpha, beta) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)]:
        g = duffing_aut_rhs(zeta=0.1, alpha=alpha, beta=beta)

        # Equilíbrios analíticos
        eqs = [np.array([0.0, 0.0])]
        if alpha*beta < 0:
            xstar = math.sqrt(abs(alpha/beta))
            eqs += [np.array([ xstar, 0.0]), np.array([-xstar, 0.0])]

        # Classificação e jacobianos
        Js, classes = [], []
        for y_eq in eqs:
            J, lam, cls = classify_equilibrium(g, y_eq)
            Js.append(J); classes.append(cls)
            print(f"  α={alpha:.2f}, β={beta:.2f}  y*={np.round(y_eq,4)}  ->  {cls}  (λ={np.round(lam,4)})")

        # Domínio de plote
        xmax = max(1.0, *(abs(e[0]) for e in eqs)); vspan = 2.0
        xlim = (-1.5 - 1.2*xmax, 1.5 + 1.2*xmax); ylim = (-vspan, vspan)

        # Figura
        fig, ax = plt.subplots()
        heatmap_global_linearized_2d(ax, eqs, Js, xlim, ylim, res=240, log=True)
        labels_done = {}
        for y_eq, cls in zip(eqs, classes):
            _plot_marker(ax, y_eq[0], y_eq[1], _kind_from_cls(cls), labels_done)
        ax.set_xlabel('x'); ax.set_ylabel('x˙')
        ax.set_title(f'Duffing (α={alpha}, β={beta}) — heatmap linearizado global + equilíbrios')
        ax.legend(); ax.grid(False); plt.show()


def plot_pendulum_heatmap():
    """
    Pêndulo simples com ζ≥0: plota heatmap 2D no plano (φ, φ˙) com equilíbrios
    kπ. Para ζ>0, os mínimos 2kπ tornam-se assintoticamente estáveis (nós/focos).
    """
    print("\n(b) Pêndulo — heatmap global + equilíbrios")
    wn = 1.0
    zeta = 0.05  # ajuste: problema da lista inclui termo de amortecimento ζ
    g = pendulum_aut_rhs(wn=wn, zeta=zeta)

    # Equilíbrios em múltiplos de π
    eqs = [np.array([k*np.pi, 0.0]) for k in range(-2, 3)]

    Js, classes = [], []
    for y_eq in eqs:
        J, lam, cls = classify_equilibrium(g, y_eq)
        Js.append(J); classes.append(cls)
        print(f"  ζ={zeta:.3f}  ϕ*={y_eq[0]/np.pi:.0f}π  ->  {cls}  (λ={np.round(lam,4)})")

    # Domínio
    xlim = (-2*np.pi, 2*np.pi); ylim = (-2.0, 2.0)

    fig, ax = plt.subplots()
    heatmap_global_linearized_2d(ax, eqs, Js, xlim, ylim, res=260, log=True)
    labels_done = {}
    for y_eq, cls in zip(eqs, classes):
        _plot_marker(ax, y_eq[0], y_eq[1], _kind_from_cls(cls), labels_done)
    ax.set_xlabel('ϕ'); ax.set_ylabel('ϕ˙')
    ax.set_title('Pêndulo com amortecimento — heatmap linearizado global + equilíbrios')
    ax.legend(); ax.grid(False); plt.show()


def plot_lorenz_heatmap():
    """
    Lorenz: para vários valores de ρ, plota uma fatia 2D (x×y) do heatmap
    linearizado e marca os equilíbrios correspondentes.
    """
    print("\n(c) Lorenz — heatmap global (fatia em x–y) + equilíbrios")
    sigma, beta = 10.0, 8.0/3.0
    for rho in [0.5, 20.0, 28.0]:
        g = lorenz_rhs(sigma=sigma, rho=rho, beta=beta)

        # Equilíbrios analíticos
        eqs = [np.array([0.0, 0.0, 0.0])]
        if rho > 1.0:
            r = math.sqrt(beta*(rho - 1.0))
            eqs += [np.array([ r,  r, rho - 1.0]), np.array([-r, -r, rho - 1.0])]

        # Classificação
        Js, classes = [], []
        for y_eq in eqs:
            J, lam, cls = classify_equilibrium(g, y_eq)
            Js.append(J); classes.append(cls)
            print(f"  ρ={rho:.2f}  y*={np.round(y_eq,4)}  ->  {cls}  (λ={np.round(lam,4)})")

        # Domínio 2D (x,y) abrangendo todos
        xmax = max(1.0, *(abs(e[0]) for e in eqs)); span = 3.0 + 1.0*xmax
        xlim = (-span, span); ylim = (-span, span)

        fig, ax = plt.subplots()
        heatmap_global_lorenz_slice(ax, eqs, Js, xlim, ylim, res=240, log=True)
        labels_done = {}
        for y_eq, cls in zip(eqs, classes):
            _plot_marker(ax, y_eq[0], y_eq[1], _kind_from_cls(cls), labels_done)  # projeção (x,y)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Lorenz (ρ={rho}) — heatmap linearizado global (fatia z=z*) + equilíbrios')
        ax.legend(); ax.grid(False); plt.show()


def plot_2gl_heatmap():
    """
    Sistema 2‑GL: busca de equilíbrios por Newton em (x1, v1=0, x2, v2=0)
    e visualização do heatmap projetado no plano (x1×x2).
    """
    print("\n(d) 2‑GL — heatmap global (plano x1×x2) + equilíbrios")
    pars = dict(z1=0.05, z2=0.05, a1=-1.2, b1=1.0, a2=-0.8, b2=1.0, rho=1.0, Os=1.0)
    g = multi2dof_rhs(**pars)

    # Busca de equilíbrios (fixa-se v1=v2=0 na caixa de posições)
    box = [(-2.0, 2.0), (0.0, 0.0), (-2.0, 2.0), (0.0, 0.0)]
    eqs = scan_equilibria(g, box, grid_per_dim=9, tol_root=1e-10)
    if len(eqs) == 0:
        print("  Nenhum equilíbrio encontrado na caixa informada.")
        return

    Js, classes = [], []
    for y_eq in eqs:
        J, lam, cls = classify_equilibrium(g, y_eq)
        Js.append(J); classes.append(cls)
        print(f"  y*={np.round(y_eq,5)}  ->  {cls}  (λ={np.round(lam,4)})")

    # Domínio em (x1,x2)
    x1max = max(1.0, *(abs(e[0]) for e in eqs))
    x2max = max(1.0, *(abs(e[2]) for e in eqs))
    span = 1.8 + 0.7*max(x1max, x2max)
    x1lim = (-span, span); x2lim = (-span, span)

    fig, ax = plt.subplots()
    heatmap_global_2gl(ax, eqs, Js, x1lim, x2lim, res=240, log=True)
    labels_done = {}
    for y_eq, cls in zip(eqs, classes):
        _plot_marker(ax, y_eq[0], y_eq[2], _kind_from_cls(cls), labels_done)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_title('2‑GL — heatmap linearizado global (x₁×x₂) + equilíbrios')
    ax.legend(); ax.grid(False); plt.show()

# =============================================================================
# 6) Execução (somente heatmaps)
# =============================================================================
if __name__ == "__main__":
    plot_duffing_heatmap()
    plot_pendulum_heatmap()   # agora com ζ>0, conforme o enunciado
    plot_lorenz_heatmap()
    plot_2gl_heatmap()
