"""
HTFRS – Healthy Time-aware Food Recommender System
Replica de 3 notebooks (Entrega Intermedia):
  Fase A: HTFRS_TimeAware_CF.ipynb        → CF con similitud Pearson ponderada por tiempo
  Fase B: HTFRS_Ingredients_Clusters.ipynb→ Similitud ingredientes + Louvain + predicción food
  Fase C: HTFRS_Full_Sellos.ipynb         → Ensamblaje y análisis de sensibilidad γ

Dependencias: pandas, numpy, scipy, sklearn, networkx, python-louvain
  pip install python-louvain networkx

Salida: Tablas_graficos/
Tiempo estimado de ejecución: 45–90 min (Fase A ~25 min, Fase B ~20 min)
Los resultados intermedios se cachean en Tablas_graficos/cache_*.pkl
"""

import os, re, sys, time, pickle, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..'))
OUT_DIR  = os.path.join(BASE_DIR, 'Tablas_graficos')
os.makedirs(OUT_DIR, exist_ok=True)

from post_processor import RecommenderPostProcessor
pp = RecommenderPostProcessor()

def _find_file(filename, extra_dirs=()):
    import glob
    candidates = list(extra_dirs) + [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, '..', filename),
        os.path.join(BASE_DIR, '..', '..', filename),
        os.path.join(BASE_DIR, '..', '..', '..', filename),
    ]
    for p in candidates:
        matches = glob.glob(p)
        if matches:
            return matches[0]
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No se encontró {filename}. Colócalo junto al script o en una carpeta padre.'
    )

REVIEWS_PATH = _find_file('reviews.csv', extra_dirs=[
    os.path.expanduser('~/.cache/kagglehub/datasets/irkaal/foodcom-recipes-and-reviews/versions/*/reviews.csv'),
])
RECIPES_PATH = _find_file('recipes_final_consolidado.csv')
print(f'reviews.csv      → {REVIEWS_PATH}')
print(f'recipes_final... → {RECIPES_PATH}')

CACHE_A = os.path.join(OUT_DIR, 'cache_fase_a.pkl')
CACHE_B = os.path.join(OUT_DIR, 'cache_fase_b.pkl')

# ── Parámetros del paper ──────────────────────────────────────────────────────
LAMBDA            = 2.5
K_NEIGHBORS       = 50
K_RECS            = 10
MIN_CORATINGS     = 3
TOP_RECIPES_GRAPH = 10000
TOP_NEIGHBORS_G   = 10
MIN_INGREDIENT_F  = 5
BETA              = 0.6
GAMMAS            = [0, 0.1, 0.2, 0.3, 0.5, 0.8]
N_EVAL_HTFRS      = 2000   # usuarios para evaluación (muestra reproducible)
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
RELEVANCE_THRESH  = 4

# ── Métricas ──────────────────────────────────────────────────────────────────
def _dcg(rel, k):
    r = np.asarray(rel, dtype=float)[:k]
    return np.sum((2**r-1)/np.log2(np.arange(2, r.size+2))) if r.size else 0.

def ndcg_at_k(rec, relevant, k=K_RECS):
    rel = [1 if r in relevant else 0 for r in rec[:k]]
    d, i = _dcg(rel, k), _dcg(sorted(rel, reverse=True), k)
    return d/i if i > 0 else 0.

def map_at_k(rec, relevant, k=K_RECS):
    hits, s, rel_set = 0, 0., set(relevant)
    for i, item in enumerate(rec[:k]):
        if item in rel_set:
            hits += 1
            s += hits/(i+1)
    return s/min(len(rel_set), k) if rel_set else 0.

def sellos_at_k(rec, sellos_dict, k=K_RECS):
    return float(np.mean([sellos_dict.get(r, 0) for r in rec[:k]]))

def sello_free_at_k(rec, sellos_dict, k=K_RECS):
    return float(np.mean([1 if sellos_dict.get(r, 0) == 0 else 0 for r in rec[:k]]))

def get_relevant(test_df, user_id):
    ut = test_df[test_df['AuthorId'] == user_id]
    return set(ut.loc[ut['Rating'] >= RELEVANCE_THRESH, 'RecipeId'])

def evaluate_all(user_recs, test_df, sellos_dict, k=K_RECS):
    rows = []
    for uid, rec in user_recs.items():
        rel = get_relevant(test_df, uid)
        hits = len(set(rec[:k]) & rel)
        p  = hits / k
        r  = hits / len(rel) if rel else 0.
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.
        # Diversidad y Novedad con recomendaciones híbridas
        hist_ids  = list(train_items_per_user.get(uid, set()))
        cluster_u = pp.cluster_dominante(hist_ids, cluster_map)
        hybrid    = pp.recomendar(rec[:k], hist_ids, cluster_map,
                                  pool_saludable, cluster_u, k=k,
                                  pool_con_sello=pool_con_sello)
        nd = pp.evaluar(hybrid, hist_ids, cluster_map,
                        pool_saludable, total_clusters, cluster_u)
        rows.append({'P@K': p, 'R@K': r, 'F1@K': f1,
                     'nDCG@K':      ndcg_at_k(rec, rel, k),
                     'MAP@K':       map_at_k(rec, rel, k),
                     'S@K':         sellos_at_k(rec, sellos_dict, k),
                     'SS@K':        sello_free_at_k(rec, sellos_dict, k),
                     'Diversidad':  nd['Diversidad'],
                     'Novedad':     nd['Novedad']})
    if not rows:
        return {}
    return {kk: float(np.mean([m[kk] for m in rows])) for kk in rows[0]}

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Carga y filtrado de datos (común a las 3 fases)
# ═══════════════════════════════════════════════════════════════════════════════
print('='*70)
print('Cargando datos...')
print('='*70)
reviews = pd.read_csv(REVIEWS_PATH)
recipes = pd.read_csv(RECIPES_PATH)
recipes['ExtractedServingSize'] = (
    recipes['ExtractedServingSize'].str.extract(r'\((.*?)\)').astype(float)
)
recipes = recipes[recipes['ExtractedServingSize'] > 0].copy()

user_counts  = reviews['AuthorId'].value_counts()
valid_users  = user_counts[user_counts > 1].index
reviews_f    = reviews[reviews['AuthorId'].isin(valid_users)].copy()
valid_rids   = set(recipes['RecipeId'])
reviews_f    = reviews_f[reviews_f['RecipeId'].isin(valid_rids)].copy()

print(f'Usuarios: {reviews_f["AuthorId"].nunique()} | '
      f'Recetas: {reviews_f["RecipeId"].nunique()} | '
      f'Ratings: {len(reviews_f)}')

# ── Corrección serving size (batches multi-unidad con ConsolidatedServings=1) ─
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from post_processor import fix_serving_sizes as _fss
recipes = _fss(recipes)

# Sellos Ley 20.606 (por 100 g)
sv = recipes['ExtractedServingSize']
recipes['IsHighCalories']     = (recipes['Calories']            / sv * 100) >= 275
recipes['IsHighSugar']        = (recipes['SugarContent']        / sv * 100) >= 10
recipes['IsHighSaturatedFat'] = (recipes['SaturatedFatContent'] / sv * 100) >= 4
recipes['IsHighSodium']       = (recipes['SodiumContent']       / sv * 100) >= 400
recipes['num_sellos']         = recipes[
    ['IsHighCalories','IsHighSugar','IsHighSaturatedFat','IsHighSodium']
].sum(axis=1).astype(int)
recipe_sellos_dict = dict(zip(recipes['RecipeId'], recipes['num_sellos']))
recipe_hf          = {rid: 1.0 - ns/4.0 for rid, ns in recipe_sellos_dict.items()}
cluster_map        = dict(zip(recipes['RecipeId'], recipes['RecipeCategory'].fillna('Unknown')))
pool_saludable     = list(recipes.loc[recipes['num_sellos'] == 0, 'RecipeId'])
pool_con_sello     = set(cluster_map.keys()) - set(pool_saludable)
total_clusters     = recipes['RecipeCategory'].nunique()

# Split 80/20 (igual en las 3 fases)
train_df, test_df = train_test_split(reviews_f, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_items_per_user: dict[int, set] = defaultdict(set)
for row in train_df[['AuthorId','RecipeId']].itertuples(index=False):
    train_items_per_user[row[0]].add(row[1])
global_mean = float(train_df['Rating'].mean())
print(f'Train: {len(train_df):,} | Test: {len(test_df):,}')

# ═══════════════════════════════════════════════════════════════════════════════
# FASE A: Time-Aware Collaborative Filtering (Eqs. 1-3)
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(CACHE_A):
    print('\n' + '='*70)
    print('Fase A: Cargando desde caché...')
    print('='*70)
    with open(CACHE_A, 'rb') as f:
        cache_a = pickle.load(f)
    user_recommendations_cf = cache_a['user_recommendations_cf']
    user_mean_rating        = cache_a['user_mean_rating']
    metrics_cf              = cache_a['metrics_cf']
    print(f'CF cargado: {len(user_recommendations_cf)} usuarios')
else:
    print('\n' + '='*70)
    print('Fase A: Time-Aware CF (Eqs. 1-3)  |  λ={} K_vecinos={}'.format(LAMBDA, K_NEIGHBORS))
    print('='*70)

    # Estructuras de datos
    train_df_copy = train_df.copy()
    train_df_copy['DateSubmitted'] = pd.to_datetime(train_df_copy['DateSubmitted'], errors='coerce')
    train_df_copy = train_df_copy.dropna(subset=['DateSubmitted'])
    date_min = train_df_copy['DateSubmitted'].min()
    date_max = train_df_copy['DateSubmitted'].max()
    train_df_copy['t_months'] = (train_df_copy['DateSubmitted'] - date_min).dt.days / 30.0
    TP = (date_max - date_min).days / 30.0
    print(f'Rango temporal: {date_min.date()} – {date_max.date()} ({TP:.1f} meses)')

    user_ratings: dict = defaultdict(dict)   # {user: {item: (rating, t_months)}}
    for row in train_df_copy[['AuthorId','RecipeId','Rating','t_months']].itertuples(index=False):
        user_ratings[row[0]][row[1]] = (row[2], row[3])

    user_mean_rating = {u: float(np.mean([r for r, t in items.values()]))
                        for u, items in user_ratings.items()}

    item_to_users: dict = defaultdict(set)
    for u, items in user_ratings.items():
        for item_id in items:
            item_to_users[item_id].add(u)

    def time_weight(t_ui, t_vi, TP, lam=LAMBDA):
        return np.exp(-lam*(TP-t_ui)/TP) * np.exp(-lam*(TP-t_vi)/TP)

    def user_similarity_tw(u, v):
        iu, iv = user_ratings[u], user_ratings[v]
        common = set(iu) & set(iv)
        if len(common) < MIN_CORATINGS:
            return 0.
        mu, mv = user_mean_rating[u], user_mean_rating[v]
        num = den_u = den_v = 0.
        for item in common:
            r_ui, t_ui = iu[item]
            r_vi, t_vi = iv[item]
            tw = time_weight(t_ui, t_vi, TP)
            du, dv = r_ui-mu, r_vi-mv
            num += du*dv*tw
            den_u += du**2*tw
            den_v += dv**2*tw
        if den_u == 0 or den_v == 0:
            return 0.
        return num / (np.sqrt(den_u)*np.sqrt(den_v))

    def find_neighbors(target):
        candidates = set()
        for item_id in user_ratings[target]:
            if item_id in item_to_users:
                candidates.update(item_to_users[item_id])
        candidates.discard(target)
        sims = []
        for c in candidates:
            sim = user_similarity_tw(target, c)
            if sim != 0:
                sims.append((c, sim))
        sims.sort(key=lambda x: abs(x[1]), reverse=True)
        return dict(sims[:K_NEIGHBORS])

    print(f'Calculando vecinos para {len(user_ratings)} usuarios...')
    t0 = time.time()
    user_neighbors = {}
    all_u = sorted(user_ratings.keys())
    for i, u in enumerate(all_u):
        user_neighbors[u] = find_neighbors(u)
        if (i+1) % 1000 == 0:
            elapsed = time.time()-t0
            rate = (i+1)/elapsed
            print(f'  {i+1}/{len(all_u)} ({rate:.1f} users/s, '
                  f'~{(len(all_u)-i-1)/rate/60:.0f} min restantes)')
    print(f'Vecinos calculados en {(time.time()-t0)/60:.1f} min')
    print(f'Usuarios con ≥1 vecino: {sum(1 for v in user_neighbors.values() if v)}')

    # CF: pool top-5000 ítems populares
    item_popularity   = train_df['RecipeId'].value_counts()
    candidate_items   = list(item_popularity.head(5000).index)
    test_users_cf     = set(test_df['AuthorId'].unique()) & set(user_neighbors.keys())

    def predict_cf(uid, item_id):
        if uid not in user_neighbors or not user_neighbors[uid]:
            return global_mean
        mu = user_mean_rating.get(uid, global_mean)
        num = den = 0.
        for v, sim in user_neighbors[uid].items():
            if v not in user_ratings or item_id not in user_ratings[v]:
                continue
            r_iv, _ = user_ratings[v][item_id]
            mv = user_mean_rating.get(v, global_mean)
            num += sim*(r_iv-mv)
            den += abs(sim)
        return float(np.clip(mu + num/den if den > 0 else mu, 0, 5))

    print(f'Generando recomendaciones CF para {len(test_users_cf)} usuarios...')
    t0 = time.time()
    user_recommendations_cf = {}
    for i, u in enumerate(sorted(test_users_cf)):
        seen  = train_items_per_user.get(u, set())
        cands = [it for it in candidate_items if it not in seen]
        preds = {it: predict_cf(u, it) for it in cands}
        user_recommendations_cf[u] = [k for k, _ in sorted(preds.items(), key=lambda x: x[1], reverse=True)[:K_RECS]]
        if (i+1) % 500 == 0:
            elapsed = time.time()-t0
            rate = (i+1)/elapsed
            print(f'  {i+1}/{len(test_users_cf)} ({rate:.1f} users/s, '
                  f'~{(len(test_users_cf)-i-1)/rate/60:.0f} min restantes)')
    print(f'Recomendaciones CF generadas en {(time.time()-t0)/60:.1f} min')

    metrics_cf = evaluate_all(user_recommendations_cf, test_df, recipe_sellos_dict)
    print('\n=== Fase A – Time-Aware CF ===')
    for m, v in metrics_cf.items():
        print(f'  {m}: {v:.6f}')

    with open(CACHE_A, 'wb') as f:
        pickle.dump({'user_recommendations_cf': user_recommendations_cf,
                     'user_mean_rating': user_mean_rating,
                     'metrics_cf': metrics_cf}, f)
    print(f'Caché Fase A guardado: {CACHE_A}')

# ═══════════════════════════════════════════════════════════════════════════════
# FASE B: Ingredient Similarity + Louvain + Food Predictions (Eqs. 4-10)
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(CACHE_B):
    print('\n' + '='*70)
    print('Fase B: Cargando desde caché...')
    print('='*70)
    with open(CACHE_B, 'rb') as f:
        cache_b = pickle.load(f)
    user_recommendations_food = cache_b['user_recommendations_food']
    user_predictions_food     = cache_b['user_predictions_food']
    item_mean_rating          = cache_b['item_mean_rating']
    metrics_food              = cache_b['metrics_food']
    print(f'Food cargado: {len(user_recommendations_food)} usuarios')
else:
    print('\n' + '='*70)
    print('Fase B: Ingredient Clusters + Louvain (Eqs. 4-10)')
    print('='*70)

    try:
        import networkx as nx
        from community import community_louvain
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-louvain', 'networkx'])
        import networkx as nx
        from community import community_louvain

    # Parseo de ingredientes
    def parse_ingredients(text):
        if pd.isna(text):
            return []
        return [m.lower().strip() for m in re.findall(r'"([^"]+)"', str(text)) if len(m.strip()) > 1]

    recipes_ing = recipes.copy()
    recipes_ing['ingredients_list'] = recipes_ing['RecipeIngredientParts'].apply(parse_ingredients)
    recipes_ing = recipes_ing[recipes_ing['ingredients_list'].str.len() > 0].copy()

    recipe_ids_ord  = recipes_ing['RecipeId'].values
    recipeid2idx_ing = {rid: i for i, rid in enumerate(recipe_ids_ord)}
    idx2recipeid_ing = {i: rid for rid, i in recipeid2idx_ing.items()}

    # Matriz binaria ingredientes
    mlb = MultiLabelBinarizer(sparse_output=True)
    ing_mat = mlb.fit_transform(recipes_ing['ingredients_list'])
    M       = ing_mat.shape[0]

    # TF-ISF weights (Eqs. 4-5)
    n_j = np.array(ing_mat.sum(axis=0)).flatten().astype(float)
    valid_mask = (n_j >= MIN_INGREDIENT_F) & (n_j <= M * 0.5)
    w_j = np.zeros(len(n_j))
    w_j[valid_mask] = np.log(M / n_j[valid_mask])
    print(f'Ingredientes válidos: {valid_mask.sum()} / {len(n_j)}')

    # Top recetas para el grafo
    recipe_popularity_b = train_df['RecipeId'].value_counts()
    top_rids_b   = set(recipe_popularity_b.head(TOP_RECIPES_GRAPH).index) & set(recipeid2idx_ing)
    top_indices  = sorted([recipeid2idx_ing[rid] for rid in top_rids_b])
    print(f'Recetas con ingredientes para grafo: {len(top_indices)}')

    # Similitud food-food (Eqs. 6-7) — submatriz ponderada
    from scipy.sparse import diags as sp_diags
    print('Calculando similitud entre recetas (puede tardar ~1 min)...')
    t0 = time.time()
    sub_ing   = ing_mat[top_indices]
    W_diag    = sp_diags(w_j)
    weighted  = sub_ing @ W_diag
    sim_mat   = (weighted @ sub_ing.T).toarray()
    print(f'Similitud calculada en {time.time()-t0:.1f}s | Shape: {sim_mat.shape}')

    local2rid = {i: idx2recipeid_ing[top_indices[i]] for i in range(len(top_indices))}
    rid2local = {rid: i for i, rid in local2rid.items()}

    # Grafo para Louvain
    G = nx.Graph()
    np.fill_diagonal(sim_mat, 0)
    for i in range(len(top_indices)):
        top_k_j = np.argsort(sim_mat[i])[-TOP_NEIGHBORS_G:]
        for j in top_k_j:
            if sim_mat[i, j] > 0:
                G.add_edge(i, j, weight=float(sim_mat[i, j]))
    print(f'Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas')

    # Louvain community detection
    partition = community_louvain.best_partition(G, random_state=RANDOM_STATE)
    cluster_to_local: dict = defaultdict(list)
    recipe_to_cluster: dict = {}
    for local_idx, cid in partition.items():
        cluster_to_local[cid].append(local_idx)
        recipe_to_cluster[local2rid[local_idx]] = cid
    print(f'Clusters: {len(cluster_to_local)} | '
          f'Tamaño avg: {np.mean([len(v) for v in cluster_to_local.values()]):.1f}')

    # Estructuras de datos para predicción food
    user_ratings_b: dict = defaultdict(dict)
    item_ratings_b: dict = defaultdict(list)
    for row in train_df[['AuthorId','RecipeId','Rating']].itertuples(index=False):
        user_ratings_b[row[0]][row[1]] = row[2]
        item_ratings_b[row[1]].append(row[2])
    item_mean_rating = {it: float(np.mean(rs)) for it, rs in item_ratings_b.items()}

    # Pre-computar datos por cluster
    cluster_data: dict = {}
    for cid, members in cluster_to_local.items():
        members_arr  = np.array(members)
        member_rids  = [local2rid[m] for m in members]
        member_means = np.array([item_mean_rating.get(rid, global_mean) for rid in member_rids])
        sim_sub      = sim_mat[np.ix_(members_arr, members_arr)]
        cluster_data[cid] = {
            'members': members_arr,
            'rids':    member_rids,
            'means':   member_means,
            'sim':     sim_sub,
        }

    # Clusters relevantes por usuario
    user_relevant_clusters: dict = defaultdict(set)
    for u, items in user_ratings_b.items():
        for rid in items:
            if rid in recipe_to_cluster:
                user_relevant_clusters[u].add(recipe_to_cluster[rid])

    test_users_b = sorted(set(test_df['AuthorId'].unique()) & set(user_ratings_b.keys()))
    print(f'\nGenerando predicciones food-based para {len(test_users_b)} usuarios...')
    t0 = time.time()

    user_recommendations_food: dict = {}
    user_predictions_food: dict     = {}

    for i, u in enumerate(test_users_b):
        rel_clusters = user_relevant_clusters.get(u, set())
        if not rel_clusters:
            continue
        seen     = train_items_per_user.get(u, set())
        u_items  = user_ratings_b.get(u, {})
        all_preds: dict = {}

        for cid in rel_clusters:
            cd    = cluster_data[cid]
            rids  = cd['rids']
            means = cd['means']
            sim_s = cd['sim']
            n     = len(rids)

            rated_mask = np.array([rid in u_items for rid in rids])
            if not rated_mask.any():
                continue

            deviations = np.zeros(n)
            for j in np.where(rated_mask)[0]:
                deviations[j] = u_items[rids[j]] - means[j]

            cand_mask = np.array([rid not in seen for rid in rids])
            cand_idx  = np.where(cand_mask)[0]
            if not len(cand_idx):
                continue

            rated_idx = np.where(rated_mask)[0]
            sim_cr    = sim_s[np.ix_(cand_idx, rated_idx)]
            devs_r    = deviations[rated_idx]
            nums      = sim_cr @ devs_r
            dens      = np.abs(sim_cr).sum(axis=1)

            for k_idx, ci in enumerate(cand_idx):
                pred = means[ci] + nums[k_idx]/dens[k_idx] if dens[k_idx] != 0 else means[ci]
                all_preds[rids[ci]] = float(np.clip(pred, 0, 5))

        if all_preds:
            sorted_items = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
            user_recommendations_food[u] = [rid for rid, _ in sorted_items[:K_RECS]]
            user_predictions_food[u]     = all_preds

        if (i+1) % 100 == 0:
            elapsed = time.time()-t0
            rate    = (i+1)/elapsed
            print(f'  {i+1}/{len(test_users_b)} ({rate:.1f} users/s, '
                  f'~{(len(test_users_b)-i-1)/rate/60:.0f} min restantes)')

    print(f'Predicciones food generadas en {(time.time()-t0)/60:.1f} min')
    print(f'Usuarios con recomendaciones food: {len(user_recommendations_food)}')

    metrics_food = evaluate_all(user_recommendations_food, test_df, recipe_sellos_dict)
    print('\n=== Fase B – Ingredient Clusters (Louvain) ===')
    for m, v in metrics_food.items():
        print(f'  {m}: {v:.6f}')

    with open(CACHE_B, 'wb') as f:
        pickle.dump({'user_recommendations_food': user_recommendations_food,
                     'user_predictions_food':     user_predictions_food,
                     'item_mean_rating':          item_mean_rating,
                     'metrics_food':              metrics_food}, f)
    print(f'Caché Fase B guardado: {CACHE_B}')

# ═══════════════════════════════════════════════════════════════════════════════
# FASE C: Full HTFRS – Ensamblaje + Análisis de Sensibilidad γ (Eqs. 11, 13)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('Fase C: HTFRS completo  |  β={}'.format(BETA))
print('='*70)

def htfrs_recommend(uid, beta=BETA, gamma=0.2, k=K_RECS):
    """
    Eq. 11: p_pref = β·p_cf + (1-β)·p_food
    Eq. 13: p_final = (1-γ)·p_pref + γ·HF·5
    """
    cf_recs    = user_recommendations_cf.get(uid, [])
    food_preds = user_predictions_food.get(uid, {})
    mean_u     = user_mean_rating.get(uid, global_mean)

    cf_scores: dict = {}
    for rank, item_id in enumerate(cf_recs):
        cf_scores[item_id] = mean_u + (len(cf_recs)-rank)/max(len(cf_recs), 1)

    all_items = set(cf_scores) | set(food_preds)
    seen      = train_items_per_user.get(uid, set())
    scores: dict = {}
    for item_id in all_items:
        if item_id in seen:
            continue
        p_cf   = cf_scores.get(item_id)
        p_food = food_preds.get(item_id)
        if p_cf is not None and p_food is not None:
            p_pref = beta*p_cf + (1-beta)*p_food
        elif p_cf is not None:
            p_pref = p_cf
        else:
            p_pref = p_food
        hf = recipe_hf.get(item_id, 0.5)
        scores[item_id] = (1-gamma)*p_pref + gamma*hf*5

    return [rid for rid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

eval_users_all = set(user_recommendations_cf.keys()) & set(test_df['AuthorId'].unique())
random.seed(RANDOM_STATE)
eval_users = set(random.sample(sorted(eval_users_all), min(N_EVAL_HTFRS, len(eval_users_all))))
print(f'Usuarios para evaluación HTFRS: {len(eval_users)} (muestra de {len(eval_users_all)})')

results_list = []
for gamma in GAMMAS:
    print(f'  Evaluando γ={gamma}...', end='', flush=True)
    t0 = time.time()
    user_recs = {}
    for u in eval_users:
        recs = htfrs_recommend(u, gamma=gamma)
        if recs:
            user_recs[u] = recs
    metrics = evaluate_all(user_recs, test_df, recipe_sellos_dict)
    metrics['gamma']   = gamma
    metrics['n_users'] = len(user_recs)
    results_list.append(metrics)
    print(f' nDCG={metrics["nDCG@K"]:.4f} | S@K={metrics["S@K"]:.4f} | '
          f'SS@K={metrics["SS@K"]:.4f}  ({time.time()-t0:.0f}s)')

results_df = pd.DataFrame(results_list)
print('\n=== Tabla de resultados HTFRS-Sellos ===')
cols_show = ['gamma','P@K','R@K','F1@K','nDCG@K','MAP@K','S@K','SS@K','Diversidad','Novedad']
print(results_df[[c for c in cols_show if c in results_df.columns]].to_string(index=False))

# ── Guardar CSV comparativo ───────────────────────────────────────────────────
comparison_rows = []
for _, row in results_df.iterrows():
    comparison_rows.append({
        'Método':    f'HTFRS-Sellos (γ={row["gamma"]})',
        'P@10':      row['P@K'],
        'R@10':      row['R@K'],
        'nDCG@10':   row['nDCG@K'],
        'S@10':      row['S@K'],
        'SS@10':     row['SS@K'],
        'Diversidad': row.get('Diversidad', None),
        'Novedad':    row.get('Novedad', None),
    })
comparison_rows.append({'Método': 'HTFRS-OMS (paper, Tabla 7)*',
                         'P@10': 0.0654, 'R@10': 0.0637, 'nDCG@10': 0.0453,
                         'S@10': None, 'SS@10': None, 'Diversidad': None, 'Novedad': None})
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(OUT_DIR, 'comparacion_htfrs_sellos.csv'), index=False)
print('\nGuardado: comparacion_htfrs_sellos.csv')

# ── Gráfico sensibilidad γ ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(results_df['gamma'], results_df['nDCG@K'], 'b-o', label='nDCG@10', linewidth=2)
ax1.plot(results_df['gamma'], results_df['P@K'],    'g-s', label='P@10',    linewidth=2)
ax1.plot(results_df['gamma'], results_df['R@K'],    'r-^', label='R@10',    linewidth=2)
ax1.set_xlabel('γ (peso del factor de salud)', fontsize=12)
ax1.set_ylabel('Métrica', fontsize=12)
ax1.set_title('HTFRS-Sellos: Relevancia vs γ', fontsize=14)
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(results_df['gamma'], results_df['S@K'],  'r-o', label='S@K (menor=mejor)',  linewidth=2)
ax2.plot(results_df['gamma'], results_df['SS@K'], 'g-s', label='SS@K (mayor=mejor)', linewidth=2)
ax2.set_xlabel('γ (peso del factor de salud)', fontsize=12)
ax2.set_ylabel('Métrica', fontsize=12)
ax2.set_title('HTFRS-Sellos: Salud vs γ', fontsize=14)
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sensitivity_gamma_sellos.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: sensitivity_gamma_sellos.png')

# ── Gráfico trade-off nDCG vs S@K ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for _, row in results_df.iterrows():
    ax.scatter(row['nDCG@K'], row['S@K'], s=150, zorder=5)
    ax.annotate(f'γ={row["gamma"]}', (row['nDCG@K'], row['S@K']),
                xytext=(8, 8), textcoords='offset points', fontsize=10)
ax.set_xlabel('nDCG@10 (relevancia, mayor=mejor)', fontsize=12)
ax.set_ylabel('S@K (sellos promedio, menor=más saludable)', fontsize=12)
ax.set_title('Trade-off Relevancia vs Salud (HTFRS-Sellos)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tradeoff_relevancia_salud.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: tradeoff_relevancia_salud.png')

# ── Gráfico barras agrupadas (γ = 0, 0.2, 0.5) ───────────────────────────────
key_gammas  = [0, 0.2, 0.5]
key_results = results_df[results_df['gamma'].isin(key_gammas)].reset_index(drop=True)
methods_lbl = [f'HTFRS-Sellos\n(γ={g})' for g in key_results['gamma']]
x     = np.arange(len(methods_lbl))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width*2, key_results['P@K'],             width, label='P@10',              color='#2196F3')
ax.bar(x - width,   key_results['R@K'],             width, label='R@10',              color='#4CAF50')
ax.bar(x,           key_results['nDCG@K'],          width, label='nDCG@10',           color='#FF9800')
ax.bar(x + width,   key_results['S@K'] / 4,        width, label='S@10/4 (norm.)',    color='#F44336')
ax.bar(x + width*2, key_results['SS@K'],            width, label='SS@10',             color='#9C27B0')
ax.set_xlabel('Modelo'); ax.set_ylabel('Valor')
ax.set_title('HTFRS-Sellos: Métricas por γ')
ax.set_xticks(x); ax.set_xticklabels(methods_lbl)
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparacion_metricas_htfrs.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: comparacion_metricas_htfrs.png')

# ── Gráfico Diversidad y Novedad vs γ ────────────────────────────────────────
if 'Diversidad' in results_df.columns and 'Novedad' in results_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(results_df['gamma'], results_df['Diversidad'], 'teal', marker='o', linewidth=2)
    axes[0].set_xlabel('γ (peso salud)', fontsize=12)
    axes[0].set_ylabel('Diversidad', fontsize=12)
    axes[0].set_title(f'Diversidad@{K_RECS} vs γ\n(clusters únicos / k)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(results_df['gamma'], results_df['Novedad'], 'darkorange', marker='s', linewidth=2)
    axes[1].set_xlabel('γ (peso salud)', fontsize=12)
    axes[1].set_ylabel('Novedad', fontsize=12)
    axes[1].set_title(f'Novedad@{K_RECS} vs γ\n(% recs nuevas, saludables y del cluster)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('HTFRS-Sellos: Diversidad y Novedad (recomendaciones híbridas) vs γ', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'htfrs_diversidad_novedad.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Guardado: htfrs_diversidad_novedad.png')

# ── Ejemplo: 10 usuarios aleatorios con recomendaciones híbridas ──────────────
print('\n' + '='*70)
print('EJEMPLO: 10 usuarios aleatorios — recomendaciones híbridas (γ=0.2)')
print('='*70)

recipe_name_map = dict(zip(
    recipes['RecipeId'],
    recipes['Name'] if 'Name' in recipes.columns else recipes['RecipeId']
))

GAMMA_EXAMPLE = 0.2
random.seed(42)
candidate_ex = list(eval_users)
sample_users = random.sample(candidate_ex, min(10, len(candidate_ex)))

for uid in sample_users:
    base_recs = htfrs_recommend(uid, gamma=GAMMA_EXAMPLE)
    hist_ids  = list(train_items_per_user.get(uid, set()))
    cluster_u = pp.cluster_dominante(hist_ids, cluster_map)
    hybrid    = pp.recomendar(base_recs, hist_ids, cluster_map,
                              pool_saludable, cluster_u, k=K_RECS,
                              pool_con_sello=pool_con_sello)
    nd        = pp.evaluar(hybrid, hist_ids, cluster_map,
                           pool_saludable, total_clusters, cluster_u)
    print(f'\nUsuario {uid} | Cluster: {cluster_u} | '
          f'Diversidad: {nd["Diversidad"]:.2f} | Novedad: {nd["Novedad"]:.2f}')
    for j, rid in enumerate(hybrid, 1):
        nm   = recipe_name_map.get(rid, rid)
        cat  = cluster_map.get(rid, '?')
        sel  = recipe_sellos_dict.get(rid, 0)
        mark = '[saludable]' if sel == 0 else f'[{int(sel)} sello(s)]'
        print(f'  {j:2d}. {str(nm)[:45]:45s} | {cat[:25]:25s} | {mark}')

# ── Resumen final ─────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESUMEN HTFRS')
print('='*70)
print(f'Fase A  – CF metrics: {metrics_cf}')
print(f'Fase B  – Food metrics: {metrics_food}')
print('\nFase C  – HTFRS-Sellos por γ:')
show_cols = [c for c in ['gamma','P@K','R@K','nDCG@K','S@K','SS@K','Diversidad','Novedad']
             if c in results_df.columns]
print(results_df[show_cols].to_string(index=False))
# ── Gráficos adicionales: Venn, Pareto, Clusters ──────────────────────────────
print('\nGenerando gráficos adicionales (Venn, Pareto, Clusters)...')

try:
    from venn import venn as _venn_fn
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, '-m', 'pip', 'install', 'venn', '-q'])
    from venn import venn as _venn_fn

# Recolectar recetas de htfrs_recommend (γ=0.2) para muestra de usuarios
random.seed(42)
_htfrs_uids = random.sample(list(eval_users), min(500, len(eval_users)))
_htfrs_rids = []
for _u in _htfrs_uids:
    _htfrs_rids.extend(htfrs_recommend(_u, gamma=0.2))

# Venn: intersección de sellos en recetas recomendadas
_htfrs_rf = recipes[recipes['RecipeId'].isin(set(_htfrs_rids))]
_hcols  = ['IsHighSaturatedFat', 'IsHighSugar', 'IsHighSodium', 'IsHighCalories']
_hnames = ['Grasas Saturadas', 'Azúcares', 'Sodio', 'Calorías']
_vsets_h = {n: set(_htfrs_rf[_htfrs_rf[c] == True]['RecipeId']) for c, n in zip(_hcols, _hnames)}
plt.figure(figsize=(10, 10))
_venn_fn(_vsets_h)
plt.title('HTFRS-Sellos — Intersección de Sellos en Recetas Recomendadas (γ=0.2)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'htfrs_venn_sellos.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: htfrs_venn_sellos.png')

# Pareto: puntos de results_df (nDCG@K vs SS@K para cada γ)
_h_pts = list(zip(
    [f'γ={row["gamma"]}' for _, row in results_df.iterrows()],
    results_df['nDCG@K'].tolist(),
    results_df['SS@K'].tolist(),
))
_h_cols = plt.cm.cool(np.linspace(0.1, 0.9, len(_h_pts)))
fig, ax = plt.subplots(figsize=(8, 6))
for (_lbl, _nd, _ss), _col in zip(_h_pts, _h_cols):
    ax.scatter([_nd], [_ss], s=160, color=_col, zorder=5, edgecolors='black', lw=1.2)
    ax.annotate(_lbl, xy=(_nd, _ss), xytext=(5, 4), textcoords='offset points', fontsize=9)
_sp2 = sorted(_h_pts, key=lambda x: x[1])
_px2, _py2, _mx2 = [], [], -1.0
for _, _nd, _ss in _sp2:
    if _ss > _mx2:
        _px2.append(_nd); _py2.append(_ss); _mx2 = _ss
if len(_px2) > 1:
    ax.plot(_px2, _py2, '--', color='navy', lw=1.8, alpha=0.7, label='Frontera Pareto')
    ax.legend(fontsize=9)
ax.set_xlabel('nDCG@10 ↑  (Relevancia)', fontsize=12)
ax.set_ylabel('SS@10 ↑  (% sin sellos)', fontsize=12)
ax.set_title('HTFRS-Sellos — Trade-off Relevancia vs Salud por γ', fontsize=12)
ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'htfrs_pareto.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: htfrs_pareto.png')

# Clusters: distribución de categorías en recetas recomendadas
_htfrs_cats = pd.Series([cluster_map.get(rid, 'Unknown') for rid in _htfrs_rids]).value_counts().head(20)
fig, ax = plt.subplots(figsize=(14, 6))
_htfrs_cats.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='white')
ax.set_title('HTFRS-Sellos — Top-20 Clusters en Recetas Recomendadas (γ=0.2)', fontsize=13)
ax.set_xlabel('Categoría (Cluster)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'htfrs_clusters.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: htfrs_clusters.png')

print('\n¡Listo!')
