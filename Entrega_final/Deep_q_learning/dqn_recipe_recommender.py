#!/usr/bin/env python3
"""
DQN Recipe Recommender — VERSIÓN MEJORADA — IIC3633 Sistemas Recomendadores
============================================================================
Mejoras sobre la versión base:
  1. Estado con embeddings SVD de usuario (20-dim) en lugar de 3 estadísticas
     genéricas → cada usuario tiene representación única de sus gustos
  2. Espacio de acciones restringido a top-10K recetas más populares →
     señal de recompensa ~10x más densa durante entrenamiento
  3. NUM_EPOCHS 15→25, N_NEXT 16→32, N_EVAL_USERS 500→2000

Instalación:
    pip install torch pandas numpy scipy scikit-learn tqdm matplotlib

Uso:
    python dqn_recipe_recommender.py                 # alpha=0.5 por defecto
    python dqn_recipe_recommender.py --alpha 0.0     # solo relevancia
    python dqn_recipe_recommender.py --alpha 1.0     # máxima penalización salud
    python dqn_recipe_recommender.py --epochs 30     # más épocas
    python dqn_recipe_recommender.py --sweep         # prueba alphas [0,.25,.5,.75,1]
    python dqn_recipe_recommender.py --examples      # mostrar ejemplos al final
"""

import os
import sys
import argparse
import random
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'Replica'))
from post_processor import RecommenderPostProcessor as _PP, fix_serving_sizes as _fss
_pp = _PP()

# ═══════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════

K             = 10
RANDOM_STATE  = 42
ALPHA_DEFAULT = 0.5

SVD_COMPONENTS = 20        # dimensiones del embedding SVD de usuario
TOP_RECIPES    = 10_000    # espacio de acciones: top-N recetas más populares
STATE_DIM      = SVD_COMPONENTS + 1   # SVD(20) + health_dim(1) = 21

HIDDEN_DIM    = 256        # capas ocultas (aumentado por mayor STATE_DIM)
LR            = 1e-3
GAMMA_RL      = 0.9
BATCH_SIZE    = 256
MEMORY_SIZE   = 300_000
NUM_EPOCHS    = 25         # más épocas para mejor convergencia
TARGET_UPDATE = 5
N_NEXT        = 32         # más muestras para estimar max Q(s',a')
N_EVAL_USERS  = 2000       # más usuarios para estimaciones estables
SCORE_BATCH   = 8_192

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    import glob

    def find_file(name, extra=()):
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = list(extra) + [
            os.path.join(base, name),
            os.path.join(base, '..', name),
            os.path.join(base, '..', '..', name),
            os.path.join(base, '..', '..', '..', name),
        ]
        for p in candidates:
            hits = glob.glob(p)
            if hits:
                return hits[0]
            if os.path.exists(p):
                return p
        return None

    rev_path = find_file('reviews.csv', extra=[
        os.path.expanduser(
            '~/.cache/kagglehub/datasets/irkaal/foodcom-recipes-and-reviews/versions/*/reviews.csv'
        )
    ])
    rec_path = find_file('recipes_final_consolidado.csv')

    if rev_path and rec_path:
        print(f"reviews  → {rev_path}")
        print(f"recipes  → {rec_path}")
        return pd.read_csv(rev_path), pd.read_csv(rec_path)

    print("Archivos no encontrados. Descargando...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
        reviews = pd.read_csv(os.path.join(path, "reviews.csv"))
    except Exception as e:
        sys.exit(f"Error descargando reviews: {e}")
    try:
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1Rl8XowC9N6cxrdiPvH4wToFUvZrrBqRG",
            "recipes_final_consolidado.csv", quiet=False,
        )
        recipes = pd.read_csv("recipes_final_consolidado.csv")
    except Exception as e:
        sys.exit(f"Error descargando recipes: {e}")
    return reviews, recipes


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(reviews, recipes):
    recipes = recipes.copy()
    if recipes["ExtractedServingSize"].dtype == object:
        extracted = recipes["ExtractedServingSize"].str.extract(r"\(([\d.]+)\)")
        recipes["ExtractedServingSize"] = pd.to_numeric(extracted[0], errors="coerce")
    recipes = recipes[recipes["ExtractedServingSize"] > 0].copy()
    recipes = _fss(recipes)

    s = recipes["ExtractedServingSize"]
    recipes["IsHighCalories"]     = ((recipes["Calories"]            / s) * 100) >= 275
    recipes["IsHighSugar"]        = ((recipes["SugarContent"]        / s) * 100) >= 10
    recipes["IsHighSaturatedFat"] = ((recipes["SaturatedFatContent"] / s) * 100) >= 4
    recipes["IsHighSodium"]       = ((recipes["SodiumContent"]       / s) * 100) >= 400
    sello_cols = ["IsHighCalories", "IsHighSugar", "IsHighSaturatedFat", "IsHighSodium"]
    recipes["num_sellos"] = recipes[sello_cols].sum(axis=1).astype(int)

    valid_users = reviews["AuthorId"].value_counts()
    valid_users = valid_users[valid_users > 1].index
    reviews = reviews[reviews["AuthorId"].isin(valid_users)].copy()
    valid_rids = set(reviews["RecipeId"].unique())
    recipes = recipes[recipes["RecipeId"].isin(valid_rids)].copy()
    reviews = reviews[reviews["RecipeId"].isin(set(recipes["RecipeId"]))].copy()

    n_u, n_r, n_i = reviews["AuthorId"].nunique(), len(recipes), len(reviews)
    print(f"Dataset: {n_u:,} usuarios | {n_r:,} recetas | {n_i:,} ratings | "
          f"densidad={n_i/(n_u*n_r)*100:.4f}%")
    return reviews, recipes


# ═══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDINGS SVD DE USUARIO  ← MEJORA 1
# ═══════════════════════════════════════════════════════════════════════════

def build_user_embeddings(reviews_train):
    """
    Factoriza la matriz usuario-item con TruncatedSVD.
    Devuelve user_factors (n_users × SVD_COMPONENTS, norma unitaria) y
    uid_to_svd que mapea AuthorId → índice en user_factors.
    """
    users = sorted(reviews_train["AuthorId"].unique())
    items = sorted(reviews_train["RecipeId"].unique())
    uid_to_idx = {u: i for i, u in enumerate(users)}
    iid_to_idx = {it: i for i, it in enumerate(items)}

    rows = reviews_train["AuthorId"].map(uid_to_idx).values
    cols = reviews_train["RecipeId"].map(iid_to_idx).values
    mat  = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )

    print(f"SVD usuario-item {len(users):,}×{len(items):,} → {SVD_COMPONENTS} componentes...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE, n_iter=7)
    user_factors = svd.fit_transform(mat).astype(np.float32)
    print(f"SVD completado en {time.time()-t0:.1f}s | var. explicada: "
          f"{svd.explained_variance_ratio_.sum()*100:.1f}%")

    # Normalizar a norma unitaria
    norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    user_factors /= norms

    uid_to_svd = {u: i for i, u in enumerate(users)}
    return user_factors, uid_to_svd


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURES DE RECETAS (ITEM)
# ═══════════════════════════════════════════════════════════════════════════

def build_item_features(recipes, reviews_train):
    """
    Tensor [n_recetas, item_dim] con features por receta.
    Features: 4 binarias sellos + num_sellos_norm + popularidad_log + rating_agg
    """
    sello_cols = ["IsHighCalories", "IsHighSugar", "IsHighSaturatedFat", "IsHighSodium"]
    parts = [recipes[sello_cols].astype(float).values]
    parts.append((recipes["num_sellos"].values / 4.0).reshape(-1, 1))

    pop = reviews_train["RecipeId"].value_counts().rename("n_rev").reset_index()
    pop.columns = ["RecipeId", "n_rev"]
    merged = recipes[["RecipeId"]].merge(pop, on="RecipeId", how="left")
    merged["n_rev"] = merged["n_rev"].fillna(0).values
    max_pop = max(merged["n_rev"].max(), 1)
    parts.append((np.log1p(merged["n_rev"].values) / np.log1p(max_pop)).reshape(-1, 1))

    if "AggregatedRating" in recipes.columns:
        ar = pd.to_numeric(recipes["AggregatedRating"], errors="coerce").fillna(0).values
        parts.append((ar / 5.0).reshape(-1, 1))

    if "FatContent" in recipes.columns:
        fc = pd.to_numeric(recipes["FatContent"], errors="coerce").fillna(0).values
        mx = max(fc.max(), 1e-6)
        parts.append((fc / mx).reshape(-1, 1))

    feat = np.hstack(parts).astype(np.float32)
    print(f"Item features: {feat.shape[0]:,} recetas × {feat.shape[1]} dims")
    return torch.FloatTensor(feat)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ESTADO DE USUARIO  ← MEJORA 1 (estado SVD personalizado)
# ═══════════════════════════════════════════════════════════════════════════

def encode_state(user_id, uid_to_svd, user_factors, avg_sellos):
    """
    Estado 21-dim: [SVD_embedding(20) | avg_sellos_norm(1)]
    El embedding SVD captura gustos individuales; avg_sellos la salud reciente.
    """
    if user_id in uid_to_svd:
        svd_vec = user_factors[uid_to_svd[user_id]]          # (20,) float32
    else:
        svd_vec = np.zeros(SVD_COMPONENTS, dtype=np.float32)
    health = np.array([min(float(avg_sellos) / 4.0, 1.0)], dtype=np.float32)
    return torch.FloatTensor(np.concatenate([svd_vec, health]))


# ═══════════════════════════════════════════════════════════════════════════
# 6. REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, maxlen=MEMORY_SIZE):
        self.buf = deque(maxlen=maxlen)

    def push(self, state, item_idx, reward, next_state):
        self.buf.append((state, item_idx, reward, next_state))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, r, ns = zip(*batch)
        return (
            torch.stack(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.stack(ns),
        )

    def __len__(self):
        return len(self.buf)


def build_replay_buffer(reviews_train, rid_to_idx, recipe_sellos, alpha,
                        user_factors, uid_to_svd):
    """
    Llena el replay buffer solo con interacciones del espacio de acciones (top recipes).
    Estado = SVD embedding del usuario + avg_sellos acumulado.
    """
    replay = ReplayBuffer(MEMORY_SIZE)

    if "DateSubmitted" in reviews_train.columns:
        df = reviews_train.sort_values("DateSubmitted")
    else:
        df = reviews_train.copy()

    # Solo interacciones con recetas dentro del espacio de acciones
    df = df[df["RecipeId"].isin(rid_to_idx)]

    user_acc = defaultdict(lambda: [0.0, 0])   # [sum_sellos, n]

    print("Construyendo replay buffer...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        uid    = row.AuthorId
        rid    = row.RecipeId
        rating = float(row.Rating)

        acc = user_acc[uid]
        avg_s = acc[0] / acc[1] if acc[1] > 0 else 0.0
        s_vec = encode_state(uid, uid_to_svd, user_factors, avg_s)

        sello_n = recipe_sellos.get(rid, 0)
        rel     = 1.0 if rating >= 4 else (0.0 if rating == 3 else -1.0)
        reward  = rel - alpha * (sello_n / 4.0)

        acc[0] += sello_n
        acc[1] += 1
        avg_s_new = acc[0] / acc[1]
        ns_vec = encode_state(uid, uid_to_svd, user_factors, avg_s_new)

        replay.push(s_vec, rid_to_idx[rid], reward, ns_vec)

    print(f"Replay buffer: {len(replay):,} experiencias (cap={MEMORY_SIZE:,})")
    return replay


# ═══════════════════════════════════════════════════════════════════════════
# 7. RED NEURONAL Q (Dueling streams)
# ═══════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Q(estado_usuario, features_receta) → score.
    Entrada: concat([state_dim, item_dim]). Salida: escalar.
    """

    def __init__(self, state_dim, item_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + item_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, item_feat):
        return self.net(torch.cat([state, item_feat], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score_all(self, state_vec, all_feats, batch=SCORE_BATCH):
        self.eval()
        n      = len(all_feats)
        scores = torch.empty(n)
        for i in range(0, n, batch):
            j  = min(i + batch, n)
            sf = all_feats[i:j].to(DEVICE)
            sv = state_vec.unsqueeze(0).expand(j - i, -1).to(DEVICE)
            scores[i:j] = self.forward(sv, sf).cpu()
        return scores


# ═══════════════════════════════════════════════════════════════════════════
# 8. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn(reviews_train, recipes_action, item_feats_action, alpha,
              user_factors, uid_to_svd):
    set_seeds()

    recipe_ids    = recipes_action["RecipeId"].tolist()
    rid_to_idx    = {rid: i for i, rid in enumerate(recipe_ids)}
    recipe_sellos = dict(zip(recipes_action["RecipeId"], recipes_action["num_sellos"]))
    n_items       = len(recipe_ids)
    item_dim      = item_feats_action.shape[1]

    replay = build_replay_buffer(
        reviews_train, rid_to_idx, recipe_sellos, alpha, user_factors, uid_to_svd
    )
    if len(replay) < BATCH_SIZE:
        sys.exit("Replay buffer demasiado pequeño.")

    q_net      = QNetwork(STATE_DIM, item_dim).to(DEVICE)
    target_net = QNetwork(STATE_DIM, item_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn   = nn.HuberLoss()

    n_steps = max(len(replay) // BATCH_SIZE, 50)

    print(f"\nEntrenamiento DQN | dispositivo={DEVICE}")
    print(f"  Acciones (top recetas): {n_items:,}  |  item_dim={item_dim}")
    print(f"  state_dim={STATE_DIM} (SVD={SVD_COMPONENTS}+health=1)")
    print(f"  Epochs={NUM_EPOCHS}  steps/epoch={n_steps:,}  alpha={alpha}\n")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        q_net.train()
        losses = []

        for _ in range(n_steps):
            states, a_idxs, rewards, next_states = replay.sample(BATCH_SIZE)
            states      = states.to(DEVICE)
            next_states = next_states.to(DEVICE)
            rewards     = rewards.to(DEVICE)

            a_feats = item_feats_action[a_idxs].to(DEVICE)
            q_cur   = q_net(states, a_feats)

            with torch.no_grad():
                samp_idx  = torch.randint(0, n_items, (BATCH_SIZE, N_NEXT))
                samp_feat = item_feats_action[samp_idx.view(-1)].view(
                    BATCH_SIZE, N_NEXT, item_dim
                ).to(DEVICE)
                ns_exp = next_states.unsqueeze(1).expand(-1, N_NEXT, -1).reshape(-1, STATE_DIM)
                sf_exp = samp_feat.reshape(-1, item_dim)
                q_ns   = target_net(ns_exp, sf_exp).view(BATCH_SIZE, N_NEXT)
                max_q  = q_ns.max(dim=1).values

            td_targets = rewards + GAMMA_RL * max_q
            loss = loss_fn(q_cur, td_targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | loss={np.mean(losses):.4f} | "
              f"{time.time()-t0:.1f}s")

    return q_net, rid_to_idx, recipe_ids


# ═══════════════════════════════════════════════════════════════════════════
# 9. EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_full(q_net, reviews_test, reviews_train, recipes_action, item_feats_action,
                  user_factors, uid_to_svd, k=K, n_users=N_EVAL_USERS,
                  cluster_map=None, pool_saludable=None, total_clusters=0):
    """
    Calcula P@K, R@K, nDCG@K, MAP@K, S@K, SS@K, Novelty, Diversity.
    Solo puntúa recetas del espacio de acciones (top-10K).
    """
    recipe_ids      = recipes_action["RecipeId"].tolist()
    rid_to_idx      = {rid: i for i, rid in enumerate(recipe_ids)}
    recipe_sellos   = dict(zip(recipes_action["RecipeId"], recipes_action["num_sellos"]))
    recipe_category = {}
    if "RecipeCategory" in recipes_action.columns:
        recipe_category = dict(zip(recipes_action["RecipeId"], recipes_action["RecipeCategory"]))

    pop_counts = reviews_train["RecipeId"].value_counts()
    total_ratings = len(reviews_train)
    pop_map = (pop_counts / total_ratings).to_dict()

    train_items = defaultdict(set)
    for row in reviews_train.itertuples(index=False):
        train_items[row.AuthorId].add(row.RecipeId)

    # Solo test sobre recetas que están en el espacio de acciones
    test_rel = defaultdict(set)
    for row in reviews_test.itertuples(index=False):
        if row.Rating >= 4 and row.RecipeId in rid_to_idx:
            test_rel[row.AuthorId].add(row.RecipeId)

    # Acumulador de salud por usuario (basado en train)
    if "DateSubmitted" in reviews_train.columns:
        df_hist = reviews_train.sort_values("DateSubmitted")
    else:
        df_hist = reviews_train
    user_health = defaultdict(lambda: [0.0, 0])
    for row in df_hist.itertuples(index=False):
        acc = user_health[row.AuthorId]
        acc[0] += recipe_sellos.get(row.RecipeId, 0)
        acc[1] += 1

    eval_users = [u for u in test_rel if len(test_rel[u]) > 0]
    np.random.shuffle(eval_users)
    eval_users = eval_users[:n_users]

    print(f"\nEvaluando {len(eval_users):,} usuarios sobre {len(recipe_ids):,} recetas...")
    t0 = time.time()

    ps, rs, ndcgs, maps_, ss, sss, novs, divs = [], [], [], [], [], [], [], []
    divs_h, novs_h = [], []
    use_hybrid = (cluster_map is not None and pool_saludable is not None)

    for uid in tqdm(eval_users, desc="Eval"):
        acc = user_health[uid]
        avg_s = acc[0] / acc[1] if acc[1] > 0 else 0.0
        sv = encode_state(uid, uid_to_svd, user_factors, avg_s)

        scores = q_net.score_all(sv, item_feats_action)

        for rid in train_items[uid]:
            if rid in rid_to_idx:
                scores[rid_to_idx[rid]] = -torch.inf

        top_k = [recipe_ids[i] for i in torch.topk(scores, k).indices.tolist()]
        rel   = test_rel[uid]
        hits  = [1 if r in rel else 0 for r in top_k]

        p    = sum(hits) / k
        r    = sum(hits) / len(rel) if rel else 0.0
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        n_hits, ap = 0, 0.0
        for i, h in enumerate(hits):
            if h:
                n_hits += 1
                ap += n_hits / (i + 1)
        map_val = ap / min(len(rel), k) if rel else 0.0

        sel = [recipe_sellos.get(rid, 0) for rid in top_k]
        ps.append(p); rs.append(r); ndcgs.append(ndcg); maps_.append(map_val)
        ss.append(float(np.mean(sel)))
        sss.append(float(np.mean([1 if s == 0 else 0 for s in sel])))

        # Novelty (self-information)
        nov = float(np.mean([-np.log2(pop_map.get(rid, 1.0/total_ratings) + 1e-12)
                             for rid in top_k]))
        novs.append(nov)

        # Diversity (categorías únicas — definición original)
        cats = [recipe_category.get(rid) for rid in top_k]
        cats = [c for c in cats if c and not (isinstance(c, float) and np.isnan(c))]
        divs.append(len(set(cats)) / len(cats) if cats else 0.0)

        # Diversidad y Novedad con post-processor (misma definición que otros modelos)
        if use_hybrid:
            hist_ids  = list(train_items[uid])
            cluster_u = _pp.cluster_dominante(hist_ids, cluster_map)
            hybrid    = _pp.recomendar(top_k, hist_ids, cluster_map,
                                       pool_saludable, cluster_u, k=k,
                                       pool_con_sello=pool_con_sello)
            nd = _pp.evaluar(hybrid, hist_ids, cluster_map,
                             pool_saludable, total_clusters, cluster_u)
            divs_h.append(nd['Diversidad'])
            novs_h.append(nd['Novedad'])

    print(f"Evaluación completada en {time.time()-t0:.1f}s")
    result = {
        "P@10":      float(np.mean(ps)),
        "R@10":      float(np.mean(rs)),
        "nDCG@10":   float(np.mean(ndcgs)),
        "MAP@10":    float(np.mean(maps_)),
        "S@10":      float(np.mean(ss)),
        "SS@10":     float(np.mean(sss)),
        "Novelty":   float(np.mean(novs)),
        "Diversity": float(np.mean(divs)),
    }
    if use_hybrid:
        result["Diversidad"] = float(np.mean(divs_h))
        result["Novedad"]    = float(np.mean(novs_h))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 10. SWEEP DE ALPHAS Y GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

SWEEP_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

BASELINES = [
    {"label": "QL† α=0",       "ndcg": 0.0281, "ss10": 0.289, "model": "Q-Learning", "biased": True},
    {"label": "QL† α=0.5",     "ndcg": 0.0265, "ss10": 0.658, "model": "Q-Learning", "biased": True},
    {"label": "HTFRS γ=0",     "ndcg": 0.0130, "ss10": 0.414, "model": "HTFRS",      "biased": False},
    {"label": "HTFRS γ=0.2",   "ndcg": 0.0098, "ss10": 0.797, "model": "HTFRS",      "biased": False},
    {"label": "HTFRS γ=0.5",   "ndcg": 0.0075, "ss10": 0.874, "model": "HTFRS",      "biased": False},
    {"label": "LightFM base",   "ndcg": 0.0033, "ss10": 0.404, "model": "LightFM",    "biased": False},
    {"label": "LightFM+sellos", "ndcg": 0.0016, "ss10": 0.575, "model": "LightFM",    "biased": False},
    {"label": "Content-Based",  "ndcg": 0.0010, "ss10": 0.432, "model": "Content",    "biased": False},
]


def run_sweep(reviews_train, reviews_test, recipes_action, item_feats_action,
              user_factors, uid_to_svd,
              alphas=SWEEP_ALPHAS, sweep_epochs=15, n_eval=N_EVAL_USERS, out_dir=".",
              cluster_map=None, pool_saludable=None, total_clusters=0):
    global NUM_EPOCHS
    original_epochs = NUM_EPOCHS
    NUM_EPOCHS = sweep_epochs

    all_results = []
    item_dim    = item_feats_action.shape[1]

    for alpha in alphas:
        print(f"\n{'#'*60}")
        print(f"# SWEEP  alpha={alpha}")
        print(f"{'#'*60}")

        model_path = os.path.join(out_dir, f"dqn_alpha_{alpha:.2f}.pt")
        q_net = QNetwork(STATE_DIM, item_dim).to(DEVICE)

        loaded = False
        if os.path.exists(model_path):
            try:
                q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
                q_net.eval()
                print(f"Modelo cargado: {model_path}")
                loaded = True
            except Exception:
                print(f"Modelo incompatible (arquitectura cambió) — reentrenando")

        if not loaded:
            q_net, _, _ = train_dqn(
                reviews_train, recipes_action, item_feats_action, alpha,
                user_factors, uid_to_svd
            )
            torch.save(q_net.state_dict(), model_path)
            print(f"Modelo guardado: {model_path}")

        res = evaluate_full(
            q_net, reviews_test, reviews_train, recipes_action, item_feats_action,
            user_factors, uid_to_svd, n_users=n_eval,
            cluster_map=cluster_map, pool_saludable=pool_saludable,
            total_clusters=total_clusters,
        )
        res["alpha"] = alpha
        all_results.append(res)
        print(f"  nDCG={res['nDCG@10']:.4f}  S@10={res['S@10']:.3f}  "
              f"SS@10={res['SS@10']:.3f}")

    NUM_EPOCHS = original_epochs
    results_df = pd.DataFrame(all_results).set_index("alpha")
    csv_path   = os.path.join(out_dir, "resultados_dqn_sweep.csv")
    results_df.to_csv(csv_path)
    print(f"\nResultados guardados: {csv_path}")
    return results_df


def plot_metrics_vs_alpha(results_df, out_dir="."):
    alphas = results_df.index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"DQN Mejorado: Métricas vs Alpha (top-{TOP_RECIPES//1000}K recetas, "
                 f"estado SVD-{SVD_COMPONENTS}dim)", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(alphas, results_df["nDCG@10"], "o-", color="#1976D2", lw=2, ms=8, label="nDCG@10")
    ax.plot(alphas, results_df["P@10"],    "s--", color="#42A5F5", lw=1.5, ms=6, label="P@10")
    ax.plot(alphas, results_df["R@10"],    "^--", color="#90CAF9", lw=1.5, ms=6, label="R@10")
    ax.set_xlabel("Alpha (α)"); ax.set_ylabel("Score")
    ax.set_title("Relevancia ↑"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_xticks(alphas)

    ax = axes[1]
    ax.plot(alphas, results_df["S@10"],  "o-",  color="#E53935", lw=2, ms=8, label="S@10 ↓")
    ax.plot(alphas, results_df["SS@10"], "s--", color="#43A047", lw=2, ms=8, label="SS@10 ↑")
    ax.set_xlabel("Alpha (α)"); ax.set_ylabel("Score")
    ax.set_title("Salud"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_xticks(alphas)

    ax = axes[2]
    ax.plot(alphas, results_df["Novelty"],   "o-",  color="#7B1FA2", lw=2, ms=8, label="Novelty ↑")
    ax.plot(alphas, results_df["Diversity"], "s--", color="#F57C00", lw=2, ms=8, label="Diversity ↑")
    ax.set_xlabel("Alpha (α)"); ax.set_ylabel("Score")
    ax.set_title("Novedad y Diversidad"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_xticks(alphas)

    plt.tight_layout()
    out = os.path.join(out_dir, "dqn_metrics_vs_alpha.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado: {out}")


def plot_pareto(results_df, out_dir="."):
    color_map = {
        "Q-Learning": "#E53935", "HTFRS": "#1E88E5",
        "LightFM": "#43A047",    "Content": "#FB8C00",
        "DQN": "#7B1FA2",
    }
    _, ax = plt.subplots(figsize=(10, 7))

    for b in BASELINES:
        c  = color_map[b["model"]]
        al = 0.45 if b["biased"] else 0.85
        ax.scatter(b["ndcg"], b["ss10"], color=c, s=110, alpha=al,
                   edgecolors="black" if b["biased"] else c, linewidths=1.4, zorder=4)
        ax.annotate(b["label"], xy=(b["ndcg"], b["ss10"]),
                    xytext=(b["ndcg"] + 0.0003, b["ss10"] + 0.015),
                    fontsize=7.5, color="#444")

    for alpha, row in results_df.iterrows():
        ax.scatter(row["nDCG@10"], row["SS@10"],
                   color=color_map["DQN"], s=140, marker="D",
                   edgecolors=color_map["DQN"], linewidths=1.5, zorder=5)
        ax.annotate(f"DQN α={alpha}",
                    xy=(row["nDCG@10"], row["SS@10"]),
                    xytext=(row["nDCG@10"] + 0.0003, row["SS@10"] - 0.035),
                    fontsize=7.5, color=color_map["DQN"])

    all_pts = [{"ndcg": b["ndcg"], "ss10": b["ss10"]} for b in BASELINES if not b["biased"]]
    for alpha, row in results_df.iterrows():
        all_pts.append({"ndcg": row["nDCG@10"], "ss10": row["SS@10"]})
    all_pts.sort(key=lambda x: -x["ndcg"])
    pareto, mx = [], -1.0
    for p in all_pts:
        if p["ss10"] > mx:
            pareto.append(p); mx = p["ss10"]
    pareto.sort(key=lambda x: x["ndcg"])
    px = [p["ndcg"] for p in pareto]
    py = [p["ss10"] for p in pareto]
    ax.plot(px, py, "--", color="#5C6BC0", lw=2.2, label="Frontera Pareto")
    ax.fill_between(px, py, alpha=0.08, color="#5C6BC0")

    handles = [
        mpatches.Patch(color=color_map["Q-Learning"], label="Q-Learning (†sesgo pool-500)"),
        mpatches.Patch(color=color_map["HTFRS"],      label="HTFRS-Sellos"),
        mpatches.Patch(color=color_map["LightFM"],    label="LightFM"),
        mpatches.Patch(color=color_map["Content"],    label="Content-Based"),
        mpatches.Patch(color=color_map["DQN"],        label=f"DQN mejorado (top-{TOP_RECIPES//1000}K)"),
        mpatches.Patch(color="#5C6BC0", alpha=0.5,    label="Frontera Pareto"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_xlabel("nDCG@10 ↑  (Relevancia)", fontsize=12)
    ax.set_ylabel("SS@10 ↑  (% Recetas sin sello)", fontsize=12)
    ax.set_title(f"Frontera de Pareto: DQN Mejorado (top-{TOP_RECIPES//1000}K, SVD-{SVD_COMPONENTS}dim) "
                 f"vs Baselines", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0, top=1.05)
    plt.tight_layout()
    out = os.path.join(out_dir, "dqn_pareto_frontier.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado: {out}")


def print_results_table(results_df):
    cols = [c for c in results_df.columns]
    header = f"{'alpha':>6} | " + " | ".join(f"{c:>9}" for c in cols)
    arrows = f"{'':>6} | " + " | ".join(
        f"{'(↓)':>9}" if c in ("S@10",) else f"{'(↑)':>9}" for c in cols
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{arrows}\n{sep}")
    for alpha, row in results_df.iterrows():
        vals = " | ".join(f"{row[c]:>9.4f}" for c in cols)
        print(f"{alpha:>6.2f} | {vals}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# 11. EJEMPLOS
# ═══════════════════════════════════════════════════════════════════════════

def _plot_venn_pareto_clusters(coll_rids, pareto_pts, label, recipes_df,
                               cluster_map_d, out_dir, prefix):
    """
    Genera 3 gráficos: Venn (sellos), Pareto (nDCG vs SS@K), Clusters.
    coll_rids  : lista de recipe IDs recomendados
    pareto_pts : lista de (label_str, ndcg, ss) para cada punto del Pareto
    """
    try:
        from venn import venn as _vfn
    except ImportError:
        import subprocess as _sp
        _sp.check_call([sys.executable, '-m', 'pip', 'install', 'venn', '-q'])
        from venn import venn as _vfn

    _hcols  = ['IsHighSaturatedFat', 'IsHighSugar', 'IsHighSodium', 'IsHighCalories']
    _hnames = ['Grasas Saturadas', 'Azúcares', 'Sodio', 'Calorías']
    _rf     = recipes_df[recipes_df['RecipeId'].isin(set(coll_rids))]

    # ── Venn ──
    if len(_rf) > 0 and all(c in _rf.columns for c in _hcols):
        _sets = {n: set(_rf[_rf[c] == True]['RecipeId']) for c, n in zip(_hcols, _hnames)}
        plt.figure(figsize=(10, 10))
        _vfn(_sets)
        plt.title(f'{label} — Intersección de Sellos en Recetas Recomendadas', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{prefix}_venn_sellos.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Guardado: {prefix}_venn_sellos.png')

    # ── Pareto ──
    if pareto_pts:
        _colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pareto_pts)))
        fig, ax = plt.subplots(figsize=(8, 6))
        for (_lbl, _nd, _ss), _col in zip(pareto_pts, _colors):
            ax.scatter([_nd], [_ss], s=180, color=_col, zorder=5,
                       edgecolors='black', linewidths=1.2)
            ax.annotate(_lbl, xy=(_nd, _ss), xytext=(5, 4),
                        textcoords='offset points', fontsize=9)
        _spts = sorted(pareto_pts, key=lambda x: x[1])
        _px, _py, _mx = [], [], -1.0
        for _, _nd, _ss in _spts:
            if _ss > _mx:
                _px.append(_nd); _py.append(_ss); _mx = _ss
        if len(_px) > 1:
            ax.plot(_px, _py, '--', color='navy', lw=1.8, alpha=0.7, label='Frontera Pareto')
            ax.legend(fontsize=9)
        ax.set_xlabel('nDCG@10 ↑  (Relevancia)', fontsize=12)
        ax.set_ylabel('SS@10 ↑  (% sin sellos)', fontsize=12)
        ax.set_title(f'{label} — Trade-off Relevancia vs Salud', fontsize=12)
        ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{prefix}_pareto.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Guardado: {prefix}_pareto.png')

    # ── Clusters ──
    if cluster_map_d and coll_rids:
        _cats = pd.Series([cluster_map_d.get(rid, 'Unknown') for rid in coll_rids])
        _top  = _cats.value_counts().head(20)
        fig, ax = plt.subplots(figsize=(14, 6))
        _top.plot(kind='bar', ax=ax, color='#7B1FA2', edgecolor='white')
        ax.set_title(f'{label} — Top-20 Clusters en Recetas Recomendadas', fontsize=13)
        ax.set_xlabel('Categoría (Cluster)', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{prefix}_clusters.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Guardado: {prefix}_clusters.png')


def _collect_dqn_recs(q_net, reviews_train, reviews_test, recipes_action,
                      item_feats_action, user_factors, uid_to_svd, n=500, k=K):
    """Recoge recipe IDs recomendados por DQN para n usuarios de test."""
    recipe_ids = recipes_action['RecipeId'].tolist()
    rid_to_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    rec_sellos = dict(zip(recipes_action['RecipeId'], recipes_action['num_sellos']))

    train_items = defaultdict(set)
    for row in reviews_train.itertuples(index=False):
        train_items[row.AuthorId].add(row.RecipeId)

    user_health = defaultdict(lambda: [0.0, 0])
    df_h = reviews_train.sort_values('DateSubmitted') if 'DateSubmitted' in reviews_train.columns else reviews_train
    for row in df_h.itertuples(index=False):
        acc = user_health[row.AuthorId]
        acc[0] += rec_sellos.get(row.RecipeId, 0)
        acc[1] += 1

    test_uids = [u for u in reviews_test['AuthorId'].unique() if u in train_items]
    random.shuffle(test_uids)

    coll = []
    for uid in test_uids[:n]:
        acc   = user_health[uid]
        avg_s = acc[0] / acc[1] if acc[1] > 0 else 0.0
        sv    = encode_state(uid, uid_to_svd, user_factors, avg_s)
        scores = q_net.score_all(sv, item_feats_action)
        for rid in train_items[uid]:
            if rid in rid_to_idx:
                scores[rid_to_idx[rid]] = -torch.inf
        coll.extend([recipe_ids[i] for i in torch.topk(scores, k).indices.tolist()])
    return coll


def show_examples(q_net, reviews_train, recipes_action, item_feats_action,
                  user_factors, uid_to_svd, n_users=10, k=K,
                  cluster_map=None, pool_saludable=None, total_clusters=0):
    recipe_ids    = recipes_action["RecipeId"].tolist()
    rid_to_idx    = {rid: i for i, rid in enumerate(recipe_ids)}
    recipe_sellos = dict(zip(recipes_action["RecipeId"], recipes_action["num_sellos"]))
    recipe_name   = dict(zip(recipes_action["RecipeId"], recipes_action["Name"]))

    if "DateSubmitted" in reviews_train.columns:
        df_h = reviews_train.sort_values("DateSubmitted")
    else:
        df_h = reviews_train
    user_acc = defaultdict(lambda: [0.0, 0, set()])
    for row in df_h.itertuples(index=False):
        acc = user_acc[row.AuthorId]
        acc[0] += recipe_sellos.get(row.RecipeId, 0)
        acc[1] += 1
        acc[2].add(row.RecipeId)

    candidates = [(uid, acc) for uid, acc in user_acc.items() if acc[1] >= 5]
    if not candidates:
        return
    random.seed(42)
    random.shuffle(candidates)
    chosen = candidates[:n_users]
    use_hybrid = (cluster_map is not None and pool_saludable is not None)

    print(f"\n{'='*70}")
    print(f"EJEMPLO: {len(chosen)} usuarios aleatorios — recomendaciones híbridas")
    print(f"{'='*70}")

    for uid, acc in chosen:
        avg_s = acc[0] / acc[1]
        sv    = encode_state(uid, uid_to_svd, user_factors, avg_s)
        seen  = acc[2]

        scores = q_net.score_all(sv, item_feats_action)
        for rid in seen:
            if rid in rid_to_idx:
                scores[rid_to_idx[rid]] = -torch.inf

        top_k     = [recipe_ids[i] for i in torch.topk(scores, k).indices.tolist()]
        hist_ids  = list(seen)
        cluster_u = _pp.cluster_dominante(hist_ids, cluster_map) if use_hybrid else None
        if use_hybrid:
            hybrid = _pp.recomendar(top_k, hist_ids, cluster_map, pool_saludable, cluster_u, k=k,
                                    pool_con_sello=pool_con_sello)
            nd     = _pp.evaluar(hybrid, hist_ids, cluster_map,
                                 pool_saludable, total_clusters, cluster_u)
            recs_show = hybrid
        else:
            recs_show = top_k
            nd = {}

        print(f"\nUsuario {uid} | {acc[1]} interacciones | avg_sellos={avg_s:.2f} | "
              f"Cluster: {cluster_u} | "
              f"Diversidad: {nd.get('Diversidad', 0):.2f} | "
              f"Novedad: {nd.get('Novedad', 0):.2f}")
        print(f"{'Rank':<5} {'Receta':<42} {'Categoría':<25} {'Sellos':>6}")
        print("-" * 80)
        for rank, rid in enumerate(recs_show, 1):
            name = str(recipe_name.get(rid, f"ID:{rid}"))[:40]
            cat  = cluster_map.get(rid, '?')[:23] if cluster_map else '?'
            sel  = recipe_sellos.get(rid, 0)
            mark = '[OK]' if sel == 0 else f'[{int(sel)}s]'
            print(f"{rank:<5} {name:<42} {cat:<25} {mark:>6}")

        sel_vals = [recipe_sellos.get(r, 0) for r in recs_show]
        print(f"  S@{k}={np.mean(sel_vals):.3f}  SS@{k}="
              f"{np.mean([1 if s==0 else 0 for s in sel_vals])*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global NUM_EPOCHS, HIDDEN_DIM
    parser = argparse.ArgumentParser(description="DQN Recipe Recommender (mejorado)")
    parser.add_argument("--alpha",        type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    parser.add_argument("--hidden",       type=int,   default=HIDDEN_DIM)
    parser.add_argument("--n-eval",       type=int,   default=N_EVAL_USERS)
    parser.add_argument("--top-recipes",  type=int,   default=TOP_RECIPES,
                        help=f"Tamaño del espacio de acciones (default {TOP_RECIPES})")
    parser.add_argument("--save",         type=str,   default="dqn_model.pt")
    parser.add_argument("--load",         type=str,   default=None)
    parser.add_argument("--examples",     action="store_true")
    parser.add_argument("--sweep",        action="store_true")
    parser.add_argument("--sweep-epochs", type=int,   default=15)
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    HIDDEN_DIM = args.hidden
    top_n      = args.top_recipes
    set_seeds()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*62}")
    print(f"  DQN Mejorado | dispositivo={DEVICE}")
    print(f"  Estado: SVD-{SVD_COMPONENTS}dim + salud | Acciones: top-{top_n//1000}K recetas")
    print(f"{'='*62}\n")

    reviews, recipes = load_data()
    reviews, recipes = preprocess(reviews, recipes)
    reviews_train, reviews_test = train_test_split(
        reviews, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Split: train={len(reviews_train):,} | test={len(reviews_test):,}\n")

    # Embeddings SVD de usuario
    user_factors, uid_to_svd = build_user_embeddings(reviews_train)

    # Restricción del espacio de acciones a top-N recetas  ← MEJORA 2
    top_recipe_ids = set(
        reviews_train["RecipeId"].value_counts().head(top_n).index
    )
    recipes_action   = recipes[recipes["RecipeId"].isin(top_recipe_ids)].copy()
    print(f"\nEspacio de acciones: {len(recipes_action):,} recetas "
          f"(top-{top_n//1000}K por popularidad en train)")

    item_feats_action = build_item_features(recipes_action, reviews_train)
    item_dim          = item_feats_action.shape[1]

    # Cluster map y pool saludable (para Diversidad/Novedad estilo post-processor)
    cluster_map    = dict(zip(recipes['RecipeId'],
                              recipes['RecipeCategory'].fillna('Unknown')
                              if 'RecipeCategory' in recipes.columns
                              else ['Unknown'] * len(recipes)))
    pool_saludable = list(recipes.loc[recipes['num_sellos'] == 0, 'RecipeId'])
    pool_con_sello = set(cluster_map.keys()) - set(pool_saludable)
    total_clusters = recipes['RecipeCategory'].nunique() if 'RecipeCategory' in recipes.columns else 1
    print(f"Cluster map: {total_clusters} categorías | "
          f"Pool saludable: {len(pool_saludable):,} recetas (0 sellos)")

    # ── SWEEP ────────────────────────────────────────────────────────────
    if args.sweep:
        print(f"\nModo SWEEP: alphas={SWEEP_ALPHAS}, epochs/modelo={args.sweep_epochs}")
        results_df = run_sweep(
            reviews_train, reviews_test, recipes_action, item_feats_action,
            user_factors, uid_to_svd,
            alphas=SWEEP_ALPHAS, sweep_epochs=args.sweep_epochs,
            n_eval=args.n_eval, out_dir=out_dir,
            cluster_map=cluster_map, pool_saludable=pool_saludable,
            total_clusters=total_clusters,
        )
        print_results_table(results_df)
        plot_metrics_vs_alpha(results_df, out_dir=out_dir)
        plot_pareto(results_df, out_dir=out_dir)
        # Venn, Pareto simple, Clusters
        print('\nGenerando gráficos adicionales (Venn, Pareto, Clusters)...')
        _best_alpha = results_df['SS@10'].idxmax()
        _q_best = QNetwork(STATE_DIM, item_dim).to(DEVICE)
        _mp = os.path.join(out_dir, f'dqn_alpha_{_best_alpha:.2f}.pt')
        if os.path.exists(_mp):
            _q_best.load_state_dict(torch.load(_mp, map_location=DEVICE))
        else:
            _q_best = q_net
        _coll = _collect_dqn_recs(_q_best, reviews_train, reviews_test, recipes_action,
                                   item_feats_action, user_factors, uid_to_svd)
        _pareto_pts = [(f'α={a}', float(row['nDCG@10']), float(row['SS@10']))
                       for a, row in results_df.iterrows()]
        _plot_venn_pareto_clusters(_coll, _pareto_pts,
                                   f'DQN (top-{top_n//1000}K)', recipes,
                                   cluster_map, out_dir, 'dqn')
        return

    # ── SINGLE ALPHA ─────────────────────────────────────────────────────
    print(f"Modo single | alpha={args.alpha} | epochs={NUM_EPOCHS}")

    if args.load and os.path.exists(args.load):
        print(f"Cargando modelo desde '{args.load}'...")
        q_net = QNetwork(STATE_DIM, item_dim, args.hidden).to(DEVICE)
        try:
            q_net.load_state_dict(torch.load(args.load, map_location=DEVICE))
            q_net.eval()
        except Exception:
            print("Modelo incompatible — reentrenando")
            q_net, _, _ = train_dqn(
                reviews_train, recipes_action, item_feats_action, args.alpha,
                user_factors, uid_to_svd,
            )
    else:
        q_net, _, _ = train_dqn(
            reviews_train, recipes_action, item_feats_action, args.alpha,
            user_factors, uid_to_svd,
        )
        torch.save(q_net.state_dict(), args.save)
        print(f"Modelo guardado en '{args.save}'")

    results = evaluate_full(
        q_net, reviews_test, reviews_train, recipes_action, item_feats_action,
        user_factors, uid_to_svd, n_users=args.n_eval,
        cluster_map=cluster_map, pool_saludable=pool_saludable,
        total_clusters=total_clusters,
    )

    arrows = {"S@10": "↓", "SS@10": "↑", "P@10": "↑", "R@10": "↑",
              "nDCG@10": "↑", "MAP@10": "↑", "Novelty": "↑", "Diversity": "↑"}
    print(f"\n{'='*62}")
    print(f"  RESULTADOS — DQN α={args.alpha} | top-{top_n//1000}K recetas")
    print(f"{'='*62}")
    for metric, val in results.items():
        print(f"  {metric:<12} {arrows.get(metric,'↑')}  {val:.4f}")
    print(f"{'='*62}")

    # Venn, Pareto, Clusters para modo single-alpha
    print('\nGenerando gráficos adicionales (Venn, Pareto, Clusters)...')
    _coll_single = _collect_dqn_recs(q_net, reviews_train, reviews_test, recipes_action,
                                     item_feats_action, user_factors, uid_to_svd)
    _pareto_single = [(f'DQN α={args.alpha}', results['nDCG@10'], results['SS@10'])]
    _plot_venn_pareto_clusters(_coll_single, _pareto_single,
                               f'DQN α={args.alpha} (top-{top_n//1000}K)', recipes,
                               cluster_map, out_dir, 'dqn')

    show_examples(q_net, reviews_train, recipes_action, item_feats_action,
                  user_factors, uid_to_svd,
                  cluster_map=cluster_map, pool_saludable=pool_saludable,
                  total_clusters=total_clusters)


if __name__ == "__main__":
    main()
