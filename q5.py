# =============================================================================
# Universidade Federal do Rio de Janeiro (UFRJ) - COPPE
# Disciplina: Mecânica de Sistemas Inteligentes (COM783)
# Aluno: Matheus Schueler de Carvalho
# Data: 25/08/2025
# Questão 5 (Bônus)
# =============================================================================

import numpy as np

# =============================================================================
# 1) Integrador RK4 (passo fixo) e integração em janelas
# =============================================================================

def rk4_step(f, t, y, h):
    """
    Executa UM passo do método de Runge–Kutta clássico de 4ª ordem.

    Parâmetros
    ----------
    f : callable
        Campo vetorial f(t, y) -> dy/dt.
    t : float
        Instante atual.
    y : ndarray
        Vetor de estado no instante t.
    h : float
        Tamanho do passo.

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


def integrate_rk4(f, t0, y0, tf, dt):
    """
    Integra y' = f(t,y) de t0 a tf com passo fixo dt (RK4).

    Observação: assume-se tf > t0; dt deve ser positivo.
    O retorno t pode diferir de tf por arredondamento de ponto flutuante,
    mas é sempre igual a t0 + n*dt (n inteiro).
    """
    if dt <= 0:
        raise ValueError("dt deve ser positivo.")
    if tf < t0:
        raise ValueError("tf deve ser maior que t0 no integrador RK4 de passo fixo.")

    n = int(np.ceil((tf - t0) / dt))
    t = float(t0)
    y = np.array(y0, dtype=float)
    for _ in range(n):
        y = rk4_step(f, t, y, dt)
        t += dt
    return t, y


# =============================================================================
# 2) Mapa de Poincaré (amostragem estroboscópica)
# =============================================================================

def poincare_map(f, y0, Omega, n_transient=200, n_samples=400,
                 steps_per_T=400, return_full=False):
    """
    Constrói o mapa de Poincaré amostrando a cada período T=2π/Ω.

    Estratégia: integra-se de t a t+T com dt = T/steps_per_T (RK4),
    descartando n_transient períodos iniciais e registrando os n_samples
    seguintes. Essa amostragem “phase-locked” evita deriva de fase.

    Retorna
    -------
    P : ndarray (n_samples, m)
        Pontos y(kT) pós-transiente.
    (opcional) Ts : ndarray com tempos de amostragem,
              T  : período,
              dt : passo usado na integração.
    """
    if Omega <= 0:
        raise ValueError("Omega deve ser positivo.")
    if steps_per_T <= 0:
        raise ValueError("steps_per_T deve ser inteiro positivo.")

    T = 2*np.pi / Omega
    dt = T / float(steps_per_T)

    # aquecimento (transiente)
    t = 0.0
    y = np.array(y0, dtype=float)
    for _ in range(int(n_transient)):
        t, y = integrate_rk4(f, t, y, t+T, dt)

    # amostragem
    P = []
    Ts = []
    for _ in range(int(n_samples)):
        t, y = integrate_rk4(f, t, y, t+T, dt)
        P.append(y.copy())
        Ts.append(t)

    P = np.array(P)
    return (P, np.array(Ts), T, dt) if return_full else P


# =============================================================================
# 3) "DBSCAN-like" minimalista (clusterização por vizinhança)
# =============================================================================

def dbscan_like(points, eps=0.05, min_pts=3):
    """
    Clusteriza 'points' (N,d) com um DBSCAN minimalista.

    - Trabalha em O(N^2), suficiente para N ~ 10^3.
    - 'eps' é o raio de vizinhança.
    - 'min_pts' é o nº mínimo de vizinhos para formar núcleo de cluster.

    Retorna
    -------
    labels : ndarray (N,)
        Rótulos em {-1 (ruído), 0,1,...,K-1}.
    """
    N = len(points)
    if N == 0:
        return np.array([], dtype=int)

    UNASSIGNED = -2
    NOISE = -1

    labels = np.full(N, UNASSIGNED, dtype=int)
    visited = np.zeros(N, dtype=bool)

    # matriz de distâncias (simples e clara)
    D = np.sqrt(((points[:, None, :] - points[None, :, :])**2).sum(axis=2))

    cluster_id = 0
    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(D[i] <= eps)[0]
        if len(neighbors) < min_pts:
            labels[i] = NOISE
            continue

        # forma novo cluster e expande por conectividade
        labels[i] = cluster_id
        seeds = set(neighbors.tolist())
        if i in seeds:
            seeds.remove(i)

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                nbrs_j = np.where(D[j] <= eps)[0]
                if len(nbrs_j) >= min_pts:
                    seeds |= set(nbrs_j.tolist())

            # inclui qualquer ponto ainda não atribuído ou marcado como ruído
            if labels[j] in (UNASSIGNED, NOISE):
                labels[j] = cluster_id

        cluster_id += 1

    # converte quaisquer UNASSIGNED remanescentes em ruído
    labels[labels == UNASSIGNED] = NOISE
    return labels


def cluster_stats(points, labels):
    """
    Estatísticas por cluster: centro, raio RMS e contagem.

    Observação: supõe pontos na MESMA ESCALA usada na clusterização.
    """
    out = {}
    uniq = np.unique(labels)
    for k in uniq:
        if k < 0:  # ruído
            continue
        idx = np.where(labels == k)[0]
        pts = points[idx]
        c = pts.mean(axis=0)
        r = np.sqrt(((pts - c)**2).sum(axis=1)).mean()
        out[int(k)] = dict(center=c, radius=r, count=len(idx))
    return out


# =============================================================================
# 4) Teste 0–1 para caos (Gottwald–Melbourne, forma por correlação)
# =============================================================================

def zero_one_test_for_chaos(signal, c_vals=None, seed=123, max_frac_lag=0.1):
    """
    Retorna K \in [0,1]: ~0 para regular/quase-periódico, ~1 para caótico.

    Implementação:
    - Remove média do sinal estroboscópico x_k.
    - Para vários c ~ U(0.1, π-0.1), projeta (p,q) por somas cumulativas.
    - Calcula o MSD M(τ) = E[Δp^2 + Δq^2] e o coef. de correlação
      |corr(M(τ), τ)|. K é a mediana em c (robusto).
    """
    x = np.asarray(signal, float)
    x = x - x.mean()
    n = len(x)

    if c_vals is None:
        rng = np.random.default_rng(seed)
        c_vals = rng.uniform(0.1, np.pi - 0.1, size=64)

    Ks = []
    max_tau = max(10, int(n * max_frac_lag))
    taus = np.arange(1, max_tau + 1)

    for c in c_vals:
        cosw = np.cos(np.arange(n) * c)
        sinw = np.sin(np.arange(n) * c)
        pc = np.cumsum(x * cosw)
        qc = np.cumsum(x * sinw)

        # MSD para lags 1..max_tau
        M = []
        for tau in taus:
            dp = pc[tau:] - pc[:-tau]
            dq = qc[tau:] - qc[:-tau]
            M.append(np.mean(dp*dp + dq*dq))
        M = np.asarray(M)

        # correlação linear entre M(τ) e τ (|r| ∈ [0,1])
        tau0 = taus - taus.mean()
        M0 = M - M.mean()
        denom = np.sqrt((tau0*tau0).sum() * (M0*M0).sum())
        if denom <= 0:
            continue
        r = float((tau0 * M0).sum() / denom)
        Ks.append(abs(r))

    if len(Ks) == 0:
        return 0.0
    return float(np.median(Ks))


# =============================================================================
# 5) Classificador principal
# =============================================================================

def classify_response_via_poincare(f, y0, Omega,
                                   n_transient=200, n_samples=400, steps_per_T=400,
                                   eps=0.05, min_pts=3, kmax=8,
                                   feature_index=0):
    """
    Classifica automaticamente a resposta sob forçamento harmônico.

    Parâmetros
    ----------
    f : callable(t,y)
        RHS do sistema (periódico em t com período T=2π/Ω).
    y0 : array_like
        Condição inicial.
    Omega : float
        Frequência do forçamento.
    n_transient, n_samples, steps_per_T : int
        Controle da amostragem estroboscópica (poincare_map).
    eps : float
        Raio da vizinhança para clusterização em ESCALA PADRONIZADA.
        Valores típicos: 0.03–0.08.
    min_pts : int
        Mínimo de vizinhos para formar um cluster (núcleo).
    kmax : int
        Máximo de clusters para considerar “período-k”.
    feature_index : int
        Índice da coordenada usada no teste 0–1 (sinal estroboscópico).

    Retorna
    -------
    result : dict
        Campos:
          label  : 'period-k', 'quasiperiodic' ou 'chaotic'
          k      : ordem k (se periódico), caso contrário None
          K01    : estatística do teste 0–1 (se não periódico)
          labels : rótulos por ponto (DBSCAN caseiro)
          P      : pontos do mapa (n_samples x m)
          T      : período
          clusters : estatísticas por cluster (se periódico)
    """
    # 1) Mapa de Poincaré (amostragem estroboscópica)
    P, Ts, T, dt = poincare_map(f, y0, Omega,
                                n_transient=n_transient,
                                n_samples=n_samples,
                                steps_per_T=steps_per_T,
                                return_full=True)

    # 2) Padronização (whitening) ANTES da clusterização
    mu = P.mean(axis=0)
    sig = P.std(axis=0)
    sig[sig == 0.0] = 1.0  # evita divisão por zero
    Z = (P - mu) / sig

    # 3) Clusterização para detecção de periodicidade
    labels = dbscan_like(Z, eps=eps, min_pts=min_pts)
    valid = labels >= 0
    n_clusters = len(np.unique(labels[valid])) if np.any(valid) else 0
    n_noise = int(np.sum(labels < 0))

    if 1 <= n_clusters <= kmax:
        stats = cluster_stats(Z, labels)  # estatísticas na escala padronizada
        mean_radius = np.mean([stats[k]['radius'] for k in stats]) if stats else np.inf
        noise_frac = n_noise / max(1, len(labels))
        # Critério prático: clusters bem compactos e pouco ruído ⇒ periódico.
        if mean_radius < 0.4*eps and noise_frac < 0.15:
            return dict(label=f"period-{n_clusters}", k=n_clusters,
                        clusters=stats, labels=labels, K01=None, P=P, T=T)

    # 4) Regular (quase‑periódico) × Caótico via teste 0–1 em uma coordenada
    x_strobe = P[:, int(feature_index)]
    K01 = zero_one_test_for_chaos(x_strobe)
    label = 'chaotic' if K01 >= 0.80 else 'quasiperiodic'
    return dict(label=label, k=None, clusters=None, labels=labels, K01=K01, P=P, T=T)


# =============================================================================
# 6) Sistemas de teste (RHS com forçamento harmônico)
# =============================================================================

def duffing_forced_rhs(zeta=0.08, alpha=-1.0, beta=1.0, gamma=0.3, Omega=1.0):
    """Duffing forçado: x¨ + 2ζ x˙ + α x + β x^3 = γ sin(Ω t)  (y=[x,v])."""
    def f(t, y):
        x, v = y
        return np.array([v, gamma*np.sin(Omega*t) - 2*zeta*v - alpha*x - beta*x**3])
    return f

def pendulum_forced_rhs(zeta=0.05, wn=1.0, gamma=1.2, Omega=0.9):
    """Pêndulo forçado: φ¨ + 2ζ φ˙ + ω_n^2 sin φ = γ sin(Ω t)  (y=[φ,ω])."""
    def f(t, y):
        phi, w = y
        return np.array([w, gamma*np.sin(Omega*t) - 2*zeta*w - wn**2*np.sin(phi)])
    return f

def linear_forced_rhs(wn=1.0, gamma=1.0, Omega=1.0):
    """Linear forçado (sem amortecimento): x¨ + ω_n^2 x = γ sin(Ω t)  (y=[x,v])."""
    def f(t, y):
        x, v = y
        return np.array([v, gamma*np.sin(Omega*t) - wn**2*x])
    return f


# =============================================================================
# 7) Demonstração
# =============================================================================

if __name__ == "__main__":
    print("="*82)
    print("QUESTÃO 5 (BÔNUS) — Classificação automática via mapas de Poincaré")
    print("="*82)

    # --- Exemplo 1: Duffing forçado (ajuste γ/Ω para ver transições) ---
    Omega1 = 1.0
    f1 = duffing_forced_rhs(zeta=0.08, alpha=-1.0, beta=1.0, gamma=0.28, Omega=Omega1)
    y0_1 = np.array([0.1, 0.0])
    r1 = classify_response_via_poincare(f1, y0_1, Omega1,
                                        n_transient=200, n_samples=400, steps_per_T=400,
                                        eps=0.05, min_pts=3, kmax=8, feature_index=0)
    print("\nDuffing forçado:")
    print(f"  Classe: {r1['label']}, k={r1.get('k')}, K01={r1.get('K01')}")
    print(f"  Mapa: {len(r1['P'])} pontos; T={r1['T']:.4f}")

    # --- Exemplo 2: Pêndulo forçado ---
    Omega2 = 0.9
    f2 = pendulum_forced_rhs(zeta=0.05, wn=1.0, gamma=1.2, Omega=Omega2)
    y0_2 = np.array([0.2, 0.0])
    r2 = classify_response_via_poincare(f2, y0_2, Omega2,
                                        n_transient=200, n_samples=400, steps_per_T=400,
                                        eps=0.05, min_pts=3, kmax=8, feature_index=0)
    print("\nPêndulo forçado:")
    print(f"  Classe: {r2['label']}, k={r2.get('k')}, K01={r2.get('K01')}")
    print(f"  Mapa: {len(r2['P'])} pontos; T={r2['T']:.4f}")

    # --- Exemplo 3: Linear forçado (tende a período-1) ---
    Omega3 = 1.0
    f3 = linear_forced_rhs(wn=1.0, gamma=1.0, Omega=Omega3)
    y0_3 = np.array([0.1, 0.0])
    r3 = classify_response_via_poincare(f3, y0_3, Omega3,
                                        n_transient=200, n_samples=400, steps_per_T=400,
                                        eps=0.05, min_pts=3, kmax=8, feature_index=0)
    print("\nLinear forçado:")
    print(f"  Classe: {r3['label']}, k={r3.get('k')}, K01={r3.get('K01')}")
    print(f"  Mapa: {len(r3['P'])} pontos; T={r3['T']:.4f}")

    # --- Resumo ---
    print("\n" + "="*82)
    print(f"{'Sistema':<18} {'Classe':<16} {'K (0–1)':<10} {'Período'}")
    print("-"*60)
    def fmt(res): return (res['label'], f"{res.get('K01'):.3f}" if res.get('K01') is not None else "—", res.get('k', "—"))
    c1,k1,p1 = fmt(r1); c2,k2,p2 = fmt(r2); c3,k3,p3 = fmt(r3)
    print(f"{'Duffing':<18} {c1:<16} {k1:<10} {p1}")
    print(f"{'Pêndulo':<18} {c2:<16} {k2:<10} {p2}")
    print(f"{'Linear':<18} {c3:<16} {k3:<10} {p3}")
    print("="*82)
