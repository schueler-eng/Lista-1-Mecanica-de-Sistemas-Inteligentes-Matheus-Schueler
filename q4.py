# =============================================================================
# Universidade Federal do Rio de Janeiro (UFRJ) - COPPE
# Disciplina: Mecânica de Sistemas Inteligentes (COM783)
# Aluno: Matheus Schueler de Carvalho
# Data: 25/08/2025
# Questão 4
# =============================================================================
"""
Bacias de Atração (com detecção automática de atratores)
--------------------------------------------------------------------
Fluxo:
1) Funções do sistema dinâmico (Duffing, Pêndulo amortecido, 2‑GDL).
2) Utilidades numéricas (Newton + jacobiano numérico, classificação espectral).
3) Núcleo genérico de bacias:
   • integração em blocos com parada antecipada;
   • distância ponderada até atratores (com tratamento de ângulos).
4) (a), (b) e (d) do enunciado.
"""
# =============================================================================
# Análise de Bacias de Atração
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from numpy.linalg import eig

# Paleta e peças de legenda para “estrela + caixinha de cor”
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# =============================================================================
# 1) Sistemas dinâmicos da questão
# =============================================================================

def duffing_ode(estado, t, zeta, alpha, beta):
    """
    Oscilador de Duffing autônomo:
        x¨ + 2 ζ x˙ + α x + β x^3 = 0
    Retorna [x˙, x¨].
    """
    x, xdot = estado
    x_ddot = -2*zeta*xdot - alpha*x - beta*x**3
    return [xdot, x_ddot]


def pendulo_ode(estado, t, wn, zeta):
    """
    Pêndulo amortecido (sem forçamento):
        φ¨ + 2 ζ φ˙ + ω_n^2 sin φ = 0
    Retorna [φ˙, φ¨].
    """
    phi, phi_dot = estado
    phi_ddot = -2*zeta*phi_dot - wn**2 * np.sin(phi)
    return [phi_dot, phi_ddot]


def sistema_2gdl_ode(estado, t, zeta1, zeta2, alpha1, alpha2, beta1, beta2, rho, Omega_s):
    """
    Sistema 2‑GDL acoplado (autônomo). Estado y = [x1, v1, x2, v2].
    """
    x1, v1, x2, v2 = estado
    dx1_dt = v1
    dv1_dt = (-2*zeta1*v1 + 2*zeta2*(v2 - v1)
              - (1 + alpha1)*x1 - beta1*x1**3 + rho*Omega_s**2*(x2 - x1))
    dx2_dt = v2
    dv2_dt = (1/rho)*(-2*zeta2*(v2 - v1) - alpha2*x2 - beta2*x2**3
                      - rho*Omega_s**2*(x2 - x1))
    return [dx1_dt, dv1_dt, dx2_dt, dv2_dt]

# -----------------------------------------------------------------------------
# Equilíbrios e estabilidade
# -----------------------------------------------------------------------------

def encontrar_equilibrios_duffing(alpha, beta):
    """
    Equilíbrios do Duffing: x* = 0 e, se αβ<0, x* = ±sqrt(-α/β).
    Retorna lista de estados [x*, 0].
    """
    eqs = [np.array([0.0, 0.0])]
    if alpha * beta < 0:
        x_eq = np.sqrt(-alpha / beta)
        eqs.append(np.array([ x_eq, 0.0]))
        eqs.append(np.array([-x_eq, 0.0]))
    return eqs


def analisar_estabilidade_duffing(ponto_eq, zeta, alpha, beta):
    """
    Estabilidade local do equilíbrio do Duffing via autovalores do Jacobiano 2×2.
    """
    x_eq, _ = ponto_eq
    J = np.array([[0.0, 1.0],
                  [-(alpha + 3*beta*x_eq**2), -2*zeta]])
    lam = eig(J)[0]
    return np.all(np.real(lam) < 0)


def filtrar_equilibrios_estaveis_duffing(equilibrios, zeta, alpha, beta):
    """Seleciona apenas os equilíbrios com partes reais negativas no espectro."""
    estaveis = []
    for eq in equilibrios:
        if analisar_estabilidade_duffing(eq, zeta, alpha, beta):
            estaveis.append(eq)
    return estaveis


def encontrar_equilibrios_pendulo():
    """Para o pêndulo amortecido, apenas a origem é atrator na janela usuais."""
    return [np.array([0.0, 0.0])]


def classificar_destino_pendulo(estado_final):
    """
    Classificação do pêndulo levando em conta a periodicidade angular.
    - Retorna 0 se a CI convergiu à origem (após “wrap” para [-π, π]).
    - Retorna −1 se não convergiu ou divergiu.
    """
    if np.any(np.isnan(estado_final)) or np.any(np.isinf(estado_final)):
        return -1
    # Redução angular e pequena ponderação menor para a componente de velocidade
    phi_f = np.arctan2(np.sin(estado_final[0]), np.cos(estado_final[0]))
    st_f  = np.array([phi_f, estado_final[1]])
    dist  = np.linalg.norm(st_f * np.array([1.0, 0.3]))
    return 0 if dist < 0.5 else -1


def encontrar_equilibrios_2gdl(alpha1, alpha2, beta1, beta2, rho, Omega_s):
    """
    Equilíbrios do 2‑GDL: resolve o sistema algébrico (x1*, x2*) com fsolve
    a partir de vários chutes e remove duplicatas por tolerância.
    """
    def sistema_equilibrio(vars_):
        x1, x2 = vars_
        f1 = (1 + alpha1)*x1 + beta1*x1**3 - rho*Omega_s**2*(x2 - x1)
        f2 = alpha2*x2 + beta2*x2**3 + rho*Omega_s**2*(x2 - x1)
        return [f1, f2]

    chutes = [(-2, -2), (-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (2, 2)]
    sols = []
    for chute in chutes:
        try:
            sol = fsolve(sistema_equilibrio, chute, xtol=1e-10)
            res = np.array(sistema_equilibrio(sol))
            if np.linalg.norm(res) < 1e-8:
                if not any(np.linalg.norm(sol - s0) < 1e-6 for s0 in sols):
                    sols.append(sol)
        except Exception:
            pass

    # Converte para o estado 4D [x1, 0, x2, 0]
    return [np.array([s[0], 0.0, s[1], 0.0]) for s in sols]


def analisar_estabilidade_2gdl(ponto_eq, zeta1, zeta2, alpha1, alpha2, beta1, beta2, rho, Omega_s):
    """
    Estabilidade local para 2‑GDL via Jacobiano 4×4.
    """
    x1, v1, x2, v2 = ponto_eq  # v1=v2=0 no equilíbrio
    J = np.zeros((4, 4))
    J[0, 1] = 1.0
    J[1, 0] = -(1 + alpha1) - 3*beta1*x1**2 - rho*Omega_s**2
    J[1, 1] = -2*zeta1 - 2*zeta2
    J[1, 2] =  rho*Omega_s**2
    J[1, 3] =  2*zeta2
    J[2, 3] = 1.0
    J[3, 0] =  Omega_s**2
    J[3, 1] =  2*zeta2 / rho
    J[3, 2] = (-alpha2 - 3*beta2*x2**2 - rho*Omega_s**2) / rho
    J[3, 3] = -2*zeta2 / rho
    lam = eig(J)[0]
    return np.all(np.real(lam) < 0)


def filtrar_equilibrios_estaveis_2gdl(equilibrios, zeta1, zeta2, alpha1, alpha2, beta1, beta2, rho, Omega_s):
    """Seleciona apenas os equilíbrios estáveis para o 2‑GDL."""
    estaveis = []
    for eq in equilibrios:
        if analisar_estabilidade_2gdl(eq, zeta1, zeta2, alpha1, alpha2, beta1, beta2, rho, Omega_s):
            estaveis.append(eq)
    return estaveis

# =============================================================================
# 2) Integração e classificação de CIs
# =============================================================================

def integrar_orbita(ci, sistema_ode, params, tempo_final=50.0):
    """
    Integra a órbita até t=tempo_final usando scipy.integrate.odeint e
    retorna o estado final y(t_f).
    """
    t_span = np.linspace(0.0, tempo_final, int(tempo_final * 20))  # ~20 passos por unidade
    try:
        traj = odeint(sistema_ode, ci, t_span, args=params)
        return traj[-1]
    except Exception:
        return np.full_like(ci, np.nan)


def classificar_destino_generico(estado_final, atratores):
    """
    Classifica a CI pelo atrator mais próximo no estado final.
    Retorna índice do atrator ou −1 (não classificado).
    """
    if np.any(np.isnan(estado_final)) or np.any(np.isinf(estado_final)) or len(atratores) == 0:
        return -1
    dists = [np.linalg.norm(estado_final - a) for a in atratores]
    return int(np.argmin(dists))


def gerar_bacia_atracao(intervalo_x, intervalo_y, resolucao,
                        sistema_ode, params, atratores,
                        construir_ci_func, classificar_func=None):
    """
    Gera a matriz de rótulos de bacia em uma malha 2D do subespaço escolhido.
    Retorna (labels, (x_vals, y_vals)).
    """
    x_vals = np.linspace(intervalo_x[0], intervalo_x[1], resolucao)
    y_vals = np.linspace(intervalo_y[0], intervalo_y[1], resolucao)

    labels = np.zeros((resolucao, resolucao), dtype=int)  # rótulos inteiros
    print(f"Calculando bacia {resolucao}×{resolucao}...")
    for i, x in enumerate(x_vals):
        if i % max(1, resolucao // 10) == 0:
            print(f"  progresso: {100*i/resolucao:5.1f}%")
        for j, y in enumerate(y_vals):
            ci = construir_ci_func(x, y)
            yf = integrar_orbita(ci, sistema_ode, params)
            classe = classificar_func(yf) if classificar_func is not None else classificar_destino_generico(yf, atratores)
            labels[j, i] = classe  # [linha,coluna] com origem 'lower'
    return labels, (x_vals, y_vals)

# =============================================================================
# 3) Plote da bacia com legenda “estrela + caixa de cor”
# =============================================================================

def plotar_bacia(bacia_map, intervalos, atratores, titulo,
                 labels=("x", "y"), extrair_coords_func=None,
                 legend_only_present=True):
    """
    Plota a bacia de atração com:
      • cores fixas por classe (−1 em cinza; 0..K−1 para K atratores);
      • estrelas (★) nos pontos de equilíbrio;
      • legenda combinando (estrela + caixinha com a cor da bacia).
    """
    x_vals, y_vals = intervalos
    K = len(atratores)

    # Classes presentes (exclui −1)
    presentes = sorted(int(c) for c in np.unique(bacia_map) if c >= 0)

    # Paleta: cinza p/ −1 + K cores (tab10; se K>10, usa tab20)
    base = list(plt.cm.tab10.colors)
    if K > len(base):
        base = list(plt.cm.tab20.colors)
    color_list = ['#BDBDBD'] + base[:max(K, 1)]  # índice 0 reservado à classe -1
    cmap = ListedColormap(color_list)

    # Normalização por degraus garantindo mapeamento rótulo→cor
    bounds = np.arange(-1.5, K + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    # Imagem categórica
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(bacia_map, origin='lower',
              extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
              cmap=cmap, norm=norm, interpolation='nearest')

    # Estrelas dos equilíbrios
    for k, eq in enumerate(atratores):
        coords = extrair_coords_func(eq) if extrair_coords_func is not None else eq[:2]
        ax.plot(coords[0], coords[1], marker='*', linestyle='None',
                markersize=11, markeredgecolor='black', markerfacecolor='white',
                zorder=5)

    # Legenda: (estrela, patch-cor) por equilíbrio presente
    handles, labels_txt = [], []
    for k in range(K):
        if legend_only_present and k not in presentes:
            continue
        star  = Line2D([0], [0], marker='*', linestyle='None',
                       mfc='white', mec='black', ms=11)
        patch = Patch(facecolor=cmap(norm(k)), edgecolor='black')
        handles.append((star, patch))
        labels_txt.append(f'Equilíbrio {k+1}')

    # Classe −1 (não classificado), se houver no mapa
    if (bacia_map == -1).any():
        empty = Line2D([], [], linestyle='None')  # sem símbolo
        patch = Patch(facecolor=cmap(norm(-1)), edgecolor='black')
        handles.append((empty, patch))
        labels_txt.append('Não classificado')

    ax.legend(handles, labels_txt, handler_map={tuple: HandlerTuple(ndivide=None)},
              loc='upper right', frameon=True, title='Equilíbrio / Bacia')

    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.set_title(titulo)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

# =============================================================================
# 4) Construtores de condições iniciais e projeções
# =============================================================================

def construir_ci_duffing(x, xdot):
    return np.array([x, xdot])

def construir_ci_pendulo(phi, phi_dot):
    return np.array([phi, phi_dot])

def construir_ci_2gdl_x1x2(x1, x2):
    return np.array([x1, 0.0, x2, 0.0])

def construir_ci_2gdl_x1v1(x1, v1):
    return np.array([x1, v1, 0.0, 0.0])

def construir_ci_2gdl_x2v2(x2, v2):
    return np.array([0.0, 0.0, x2, v2])

def extrair_coords_x1x2(eq):
    return [eq[0], eq[2]]

def extrair_coords_x1v1(eq):
    return [eq[0], 0.0]  # equilíbrios têm v1=0

def extrair_coords_x2v2(eq):
    return [eq[2], 0.0]  # equilíbrios têm v2=0

def extrair_coords_pendulo(_):
    return [0.0, 0.0]

# =============================================================================
# 5) Pipeline principal
# =============================================================================

def resolver_questao_4():
    print("=== RESOLUÇÃO DA QUESTÃO 4 — BACIAS DE ATRAÇÃO ===\n")

    # ----------------------- (a) Duffing --------------------------------------
    print("1) Oscilador de Duffing")
    zeta_duff, alpha_duff, beta_duff = 0.25, -1.2, 1.5
    params_duff = (zeta_duff, alpha_duff, beta_duff)

    eqs_duff = encontrar_equilibrios_duffing(alpha_duff, beta_duff)
    atr_duff = filtrar_equilibrios_estaveis_duffing(eqs_duff, zeta_duff, alpha_duff, beta_duff)
    print(f"   atratores estáveis: {len(atr_duff)}")

    bmap_duff, (xs_d, ys_d) = gerar_bacia_atracao(
        intervalo_x=(-5, 5), intervalo_y=(-5, 5), resolucao=300,
        sistema_ode=duffing_ode, params=params_duff,
        atratores=atr_duff, construir_ci_func=construir_ci_duffing)

    plotar_bacia(bmap_duff, (xs_d, ys_d), atr_duff,
                 "Bacia de Atração — Duffing", labels=("x", "ẋ"))

    # ----------------------- (b) Pêndulo amortecido --------------------------
    print("\n2) Pêndulo amortecido")
    wn_p, zeta_p = 1.0, 0.5
    params_p = (wn_p, zeta_p)
    atr_pend = encontrar_equilibrios_pendulo()

    bmap_p, (xs_p, ys_p) = gerar_bacia_atracao(
        intervalo_x=(-np.pi, np.pi), intervalo_y=(-3.0, 3.0), resolucao=300,
        sistema_ode=pendulo_ode, params=params_p,
        atratores=atr_pend, construir_ci_func=construir_ci_pendulo,
        classificar_func=classificar_destino_pendulo)

    plotar_bacia(bmap_p, (xs_p, ys_p), atr_pend,
                 "Bacia de Atração — Pêndulo amortecido",
                 labels=("ϕ", "ϕ̇"), extrair_coords_func=extrair_coords_pendulo)

    # ----------------------- (c) Sistema 2‑GDL --------------------------------
    print("\n3) Sistema 2‑GDL acoplado")
    z1, z2 = 0.025, 0.025
    a1, a2 = -2.0, -1.0
    b1, b2 = 1.0, 1.0
    rho, Os = 0.5, 1.0
    params_2 = (z1, z2, a1, a2, b1, b2, rho, Os)

    eqs_2 = encontrar_equilibrios_2gdl(a1, a2, b1, b2, rho, Os)
    atr_2 = filtrar_equilibrios_estaveis_2gdl(eqs_2, z1, z2, a1, a2, b1, b2, rho, Os)
    print(f"   atratores estáveis: {len(atr_2)}")

    # (i) plano x1×x2
    bmap_12, (xs_12, ys_12) = gerar_bacia_atracao(
        intervalo_x=(-2.0, 2.0), intervalo_y=(-2.0, 2.0), resolucao=300,
        sistema_ode=sistema_2gdl_ode, params=params_2,
        atratores=atr_2, construir_ci_func=construir_ci_2gdl_x1x2)

    plotar_bacia(bmap_12, (xs_12, ys_12), atr_2,
                 "Bacia de Atração — 2‑GDL (x₁ × x₂)",
                 labels=("x₁", "x₂"), extrair_coords_func=extrair_coords_x1x2)

    # (ii) plano x1×ẋ1
    bmap_1v1, (xs_1v1, ys_1v1) = gerar_bacia_atracao(
        intervalo_x=(-2.0, 2.0), intervalo_y=(-2.0, 2.0), resolucao=300,
        sistema_ode=sistema_2gdl_ode, params=params_2,
        atratores=atr_2, construir_ci_func=construir_ci_2gdl_x1v1)

    plotar_bacia(bmap_1v1, (xs_1v1, ys_1v1), atr_2,
                 "Bacia de Atração — 2‑GDL (x₁ × ẋ₁)",
                 labels=("x₁", "ẋ₁"), extrair_coords_func=extrair_coords_x1v1)

    # (iii) plano x2×ẋ2
    bmap_2v2, (xs_2v2, ys_2v2) = gerar_bacia_atracao(
        intervalo_x=(-2.0, 2.0), intervalo_y=(-2.0, 2.0), resolucao=300,
        sistema_ode=sistema_2gdl_ode, params=params_2,
        atratores=atr_2, construir_ci_func=construir_ci_2gdl_x2v2)

    plotar_bacia(bmap_2v2, (xs_2v2, ys_2v2), atr_2,
                 "Bacia de Atração — 2‑GDL (x₂ × ẋ₂)",
                 labels=("x₂", "ẋ₂"), extrair_coords_func=extrair_coords_x2v2)

    print("\n=== ANÁLISE CONCLUÍDA ===")


# =============================================================================
# Execução
# =============================================================================
if __name__ == "__main__":
    resolver_questao_4()
