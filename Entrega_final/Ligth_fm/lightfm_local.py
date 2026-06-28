"""
LightFM BPR – Modelo Base y Modelo Híbrido con Sellos Nutricionales
Replica de LightFM_local.ipynb (Entrega Intermedia)
Salida: Tablas_graficos/

NOTA: LightFM tiene un bug en su extensión C en Windows (Python 3.10) con matrices
de 72K×257K: unsafe cast int32→float32 en el loop BPR causa access violation.
Este script usa lightfm si está disponible y no falla; en caso contrario usa
TruncatedSVD (scikit-learn), que es metodológicamente equivalente (factorización
de matrices latentes), para producir resultados comparables.
"""

import os
import sys
import time
import random
import platform
import faulthandler
faulthandler.enable()   # imprime traceback C-level si hay segfault
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from post_processor import RecommenderPostProcessor
pp = RecommenderPostProcessor()

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'Tablas_graficos')
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Hiperparámetros (igual al notebook) ──────────────────────────────────────
K              = 10
TEST_PERCENTAGE = 0.20
LEARNING_RATE  = 0.25
LOSS_FUNCTION  = 'bpr'
NO_COMPONENTS  = 20
NO_EPOCHS      = 20
NO_THREADS     = min(16, os.cpu_count() or 4)
ITEM_ALPHA     = 1e-6
USER_ALPHA     = 1e-6
SEED           = 42
ALPHA_HEALTH   = 0.1   # penalización salud para modelo híbrido SVD (reducido vs BPR por escala de scores SVD)

# ── 1. Carga y filtrado ───────────────────────────────────────────────────────
print('Cargando datos...')
reviews = pd.read_csv(REVIEWS_PATH)
recipes = pd.read_csv(RECIPES_PATH)
recipes['ExtractedServingSize'] = (
    recipes['ExtractedServingSize'].str.extract(r'\((.*?)\)').astype(float)
)

review_counts       = reviews['AuthorId'].value_counts()
little_review_users = review_counts[review_counts == 1]
reviews_f = reviews[~reviews['AuthorId'].isin(little_review_users.index)].copy()
recipes_f = recipes[recipes['RecipeId'].isin(reviews_f['RecipeId'].unique())].copy()
recipes_f = recipes_f[recipes_f['ExtractedServingSize'] > 0].copy()
reviews_f = reviews_f[reviews_f['RecipeId'].isin(recipes_f['RecipeId'].unique())].copy()
print(f'Reviews filtrado: {len(reviews_f):,} | Recipes filtrado: {len(recipes_f):,}')

# ── Corrección serving size (batches multi-unidad con ConsolidatedServings=1) ─
from post_processor import fix_serving_sizes as _fss
recipes_f = _fss(recipes_f)

# ── 2. Sellos Ley 20.606 ─────────────────────────────────────────────────────
sv = recipes_f['ExtractedServingSize']
recipes_f = recipes_f.copy()
recipes_f['IsHighSugar']        = (recipes_f['SugarContent']        / sv * 100) >= 10
recipes_f['IsHighSaturatedFat'] = (recipes_f['SaturatedFatContent'] / sv * 100) >= 4
recipes_f['IsHighCalories']     = (recipes_f['Calories']            / sv * 100) >= 275
recipes_f['IsHighSodium']       = (recipes_f['SodiumContent']       / sv * 100) >= 400
recipes_f['TotalSellos']        = (
    recipes_f['IsHighSugar'].astype(int) +
    recipes_f['IsHighSaturatedFat'].astype(int) +
    recipes_f['IsHighCalories'].astype(int) +
    recipes_f['IsHighSodium'].astype(int)
)
recipe_sellos_dict = dict(zip(recipes_f['RecipeId'], recipes_f['TotalSellos']))
cluster_map        = dict(zip(recipes_f['RecipeId'], recipes_f['RecipeCategory'].fillna('Unknown')))
pool_saludable     = list(recipes_f.loc[recipes_f['TotalSellos'] == 0, 'RecipeId'])
pool_con_sello     = set(cluster_map.keys()) - set(pool_saludable)
total_clusters     = recipes_f['RecipeCategory'].nunique()
print('Distribución de sellos:')
print(recipes_f['TotalSellos'].value_counts().sort_index())

# ── 3. Intento de usar LightFM; fallback a TruncatedSVD ──────────────────────
#
# LightFM BPR tiene un bug conocido en Windows (Python 3.10+) con matrices
# grandes: el loop BPR en C usa un cast inseguro int32→float32 que provoca
# un access violation (segfault) al inicio del primer epoch. El proceso muere
# sin exception Python, por eso no hay traceback visible.
# Referencia: github.com/lyst/lightfm/issues/590
#
USE_LIGHTFM = False
_ON_WINDOWS = platform.system() == 'Windows'

try:
    import lightfm
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import precision_at_k as lfm_p_at_k
    from lightfm.evaluation import recall_at_k    as lfm_r_at_k
    from lightfm.evaluation import auc_score      as lfm_auc
    if _ON_WINDOWS:
        print(f'\nLightFM {lightfm.__version__} detectado — PERO estás en Windows.')
        print('Bug conocido: access violation en fit_bpr con matrices >72K usuarios.')
        print('→ Usando TruncatedSVD como fallback (factorización matricial equivalente).')
    else:
        USE_LIGHTFM = True
        print(f'\nLightFM {lightfm.__version__} disponible – usando LightFM BPR')
except ImportError:
    print('\nLightFM no instalado – usando TruncatedSVD')

# ── Métricas comunes ──────────────────────────────────────────────────────────
def _dcg(rel, k):
    r = np.asarray(rel, dtype=float)[:k]
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2))) if r.size else 0.

def ndcg_at_k(rec, relevant, k=K):
    rel = [1 if r in relevant else 0 for r in rec[:k]]
    d, i = _dcg(rel, k), _dcg(sorted(rel, reverse=True), k)
    return d / i if i > 0 else 0.

def map_at_k(rec, relevant, k=K):
    hits, s, rel_set = 0, 0., set(relevant)
    for i, item in enumerate(rec[:k]):
        if item in rel_set:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(rel_set), k) if rel_set else 0.

def sellos_at_k(rec, k=K):
    return float(np.mean([recipe_sellos_dict.get(r, 0) for r in rec[:k]]))

def sello_free_at_k(rec, k=K):
    return float(np.mean([1 if recipe_sellos_dict.get(r, 0) == 0 else 0 for r in rec[:k]]))

# ── 4A. RAMA LIGHTFM ─────────────────────────────────────────────────────────
if USE_LIGHTFM:
    np.random.seed(SEED)

    dataset_base = Dataset()
    dataset_base.fit(users=reviews_f['AuthorId'].unique(),
                     items=reviews_f['RecipeId'].unique())
    interactions, _ = dataset_base.build_interactions(
        reviews_f[['AuthorId', 'RecipeId', 'Rating']].itertuples(index=False, name=None)
    )
    uids, iids, vals = sp.find(interactions)
    idx = np.random.permutation(len(uids))
    uids, iids, vals = uids[idx], iids[idx], vals[idx]
    cut = int((1 - TEST_PERCENTAGE) * len(uids))
    n_u, n_i = interactions.shape

    train_int = sp.coo_matrix((vals[:cut], (uids[:cut], iids[:cut])), shape=(n_u, n_i)).tocsr()
    test_int  = sp.coo_matrix((vals[cut:], (uids[cut:], iids[cut:])), shape=(n_u, n_i)).tocsr()
    uid_map, _, iid_map, _ = dataset_base.mapping()
    print(f'Train: {train_int.nnz:,} | Test: {test_int.nnz:,}')

    def _get_ranks(model, test_i, train_i, item_feat=None):
        return model.predict_rank(test_interactions=test_i, train_interactions=train_i,
                                  item_features=item_feat, num_threads=NO_THREADS)

    def eval_ndcg_map_lfm(ranks, test_i, k=K):
        ranks_csr = ranks.tocsr()
        test_csr  = test_i.tocsr()
        ndcgs, aps = [], []
        for uid in range(test_i.shape[0]):
            ti = test_csr[uid].indices
            if not len(ti):
                continue
            ur = ranks_csr[uid]
            dcg, hits_at_rank = 0., []
            for idx_ in ti:
                rank = ur[0, idx_]
                if 0 < rank <= k:
                    dcg += 1. / np.log2(rank + 1)
                    hits_at_rank.append(int(rank))
            idcg = sum(1./np.log2(i+2) for i in range(min(k, len(ti))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.)
            hits_at_rank.sort()
            ap = sum((i+1)/r for i, r in enumerate(hits_at_rank))
            n_rel = min(len(ti), k)
            aps.append(ap / n_rel if n_rel > 0 else 0.)
        return (np.mean(ndcgs) if ndcgs else 0.), (np.mean(aps) if aps else 0.)

    def eval_sellos_lfm(ranks, test_i, k=K):
        inv_iid = {v: kk for kk, v in iid_map.items()}
        ranks_csr = ranks.tocsr()
        test_csr  = test_i.tocsr()
        s_list, sf_list = [], []
        for uid in range(test_i.shape[0]):
            if test_csr[uid].nnz == 0:
                continue
            ur = np.asarray(ranks_csr[uid].todense()).flatten()
            top_k = np.where((ur > 0) & (ur <= k))[0]
            if not len(top_k):
                continue
            sellos = [recipe_sellos_dict.get(inv_iid.get(idx_, -1), 0) for idx_ in top_k]
            s_list.append(np.mean(sellos))
            sf_list.append(np.mean([1 if s == 0 else 0 for s in sellos]))
        return (np.mean(s_list) if s_list else 0.), (np.mean(sf_list) if sf_list else 0.)

    # Modelo base (sin sellos)
    print('\nEntrenando modelo base LightFM BPR...')
    t0 = time.perf_counter()
    model_base = LightFM(loss=LOSS_FUNCTION, no_components=NO_COMPONENTS,
                         learning_rate=LEARNING_RATE, item_alpha=ITEM_ALPHA,
                         user_alpha=USER_ALPHA, random_state=np.random.RandomState(SEED))
    model_base.fit(interactions=train_int, epochs=NO_EPOCHS,
                   num_threads=NO_THREADS, verbose=True)
    print(f'Entrenamiento: {time.perf_counter()-t0:.1f}s')

    t0 = time.perf_counter()
    p_sin = lfm_p_at_k(model_base, test_int, train_int, k=K, num_threads=NO_THREADS).mean()
    r_sin = lfm_r_at_k(model_base, test_int, train_int, k=K, num_threads=NO_THREADS).mean()
    auc_sin = lfm_auc(model_base, test_int, train_int, num_threads=NO_THREADS).mean()
    ranks_sin = _get_ranks(model_base, test_int, train_int)
    ndcg_sin, map_sin = eval_ndcg_map_lfm(ranks_sin, test_int)
    f1_sin = 2*p_sin*r_sin/(p_sin+r_sin) if (p_sin+r_sin) > 0 else 0.
    s_sin, ss_sin = eval_sellos_lfm(ranks_sin, test_int)
    print(f'Evaluación: {time.perf_counter()-t0:.1f}s')

    # Modelo híbrido (con sellos como item_features)
    sello_feats = ['IsHighSugar', 'IsHighSaturatedFat', 'IsHighCalories', 'IsHighSodium']
    dataset2 = Dataset()
    dataset2.fit(users=reviews_f['AuthorId'].unique(),
                 items=reviews_f['RecipeId'].unique(),
                 item_features=sello_feats)
    interactions2, _ = dataset2.build_interactions(
        reviews_f[['AuthorId', 'RecipeId', 'Rating']].itertuples(index=False, name=None)
    )
    recipes_in = recipes_f[recipes_f['RecipeId'].isin(reviews_f['RecipeId'].unique())]
    item_features_list = [(row['RecipeId'], [f for f in sello_feats if row.get(f, False)])
                          for _, row in recipes_in.iterrows()]
    item_features_list = [(rid, fs) for rid, fs in item_features_list if fs]
    item_feat2 = dataset2.build_item_features(item_features_list)
    uid_map2, _, iid_map2, _ = dataset2.mapping()

    uids2, iids2, vals2 = sp.find(interactions2)
    idx2 = np.random.permutation(len(uids2))
    uids2, iids2, vals2 = uids2[idx2], iids2[idx2], vals2[idx2]
    cut2 = int((1 - TEST_PERCENTAGE) * len(uids2))
    n_u2, n_i2 = interactions2.shape
    train2 = sp.coo_matrix((vals2[:cut2], (uids2[:cut2], iids2[:cut2])), shape=(n_u2, n_i2)).tocsr()
    test2  = sp.coo_matrix((vals2[cut2:], (uids2[cut2:], iids2[cut2:])), shape=(n_u2, n_i2)).tocsr()

    print('\nEntrenando modelo híbrido LightFM BPR + sellos...')
    t0 = time.perf_counter()
    model2 = LightFM(loss=LOSS_FUNCTION, no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE, item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA, random_state=np.random.RandomState(SEED))
    model2.fit(interactions=train2, item_features=item_feat2,
               epochs=NO_EPOCHS, num_threads=NO_THREADS, verbose=True)
    print(f'Entrenamiento: {time.perf_counter()-t0:.1f}s')

    t0 = time.perf_counter()
    p_con   = lfm_p_at_k(model2, test2, train2, k=K, item_features=item_feat2, num_threads=NO_THREADS).mean()
    r_con   = lfm_r_at_k(model2, test2, train2, k=K, item_features=item_feat2, num_threads=NO_THREADS).mean()
    auc_con = lfm_auc(model2, test2, train2, item_features=item_feat2, num_threads=NO_THREADS).mean()
    ranks_con = _get_ranks(model2, test2, train2, item_feat=item_feat2)
    ndcg_con, map_con = eval_ndcg_map_lfm(ranks_con, test2)
    f1_con = 2*p_con*r_con/(p_con+r_con) if (p_con+r_con) > 0 else 0.
    inv_iid2 = {v: kk for kk, v in iid_map2.items()}
    s_con, ss_con = eval_sellos_lfm(ranks_con, test2)
    print(f'Evaluación: {time.perf_counter()-t0:.1f}s')

# ── 4B. RAMA TRUNCATEDSVD (fallback) ─────────────────────────────────────────
else:
    from sklearn.decomposition import TruncatedSVD

    print('\nConstruyendo matriz usuario-item...')
    all_users = reviews_f['AuthorId'].unique()
    all_items = reviews_f['RecipeId'].unique()
    uid2idx   = {u: i for i, u in enumerate(all_users)}
    iid2idx   = {it: i for i, it in enumerate(all_items)}
    idx2iid   = {i: it for it, i in iid2idx.items()}
    n_u, n_i  = len(all_users), len(all_items)

    np.random.seed(SEED)
    reviews_s = reviews_f.sample(frac=1, random_state=SEED)
    cut = int((1 - TEST_PERCENTAGE) * len(reviews_s))
    train_r = reviews_s.iloc[:cut]
    test_r  = reviews_s.iloc[cut:]

    rows_tr = [uid2idx[u] for u in train_r['AuthorId']]
    cols_tr = [iid2idx[it] for it in train_r['RecipeId']]
    train_mat = sp.csr_matrix((np.ones(len(rows_tr)), (rows_tr, cols_tr)), shape=(n_u, n_i))

    rows_te = [uid2idx[u] for u in test_r['AuthorId']]
    cols_te = [iid2idx[it] for it in test_r['RecipeId']]
    vals_te = test_r['Rating'].values
    test_mat  = sp.csr_matrix((vals_te, (rows_te, cols_te)), shape=(n_u, n_i))

    # health penalty vector (normalizado)
    health_penalty = np.array([recipe_sellos_dict.get(idx2iid.get(i, -1), 0) / 4.
                                for i in range(n_i)], dtype=np.float32)

    def svd_evaluate(user_factors, item_factors, penalty=None, alpha=0., k=K, n_eval=5000):
        """Evalúa un modelo SVD en n_eval usuarios de test que tienen >= 1 item relevante."""
        eval_uidxs = []
        for uidx in range(test_mat.shape[0]):
            row = test_mat[uidx]
            if len(row.indices) > 0 and (row.data >= 4).any():
                eval_uidxs.append(uidx)
        np.random.shuffle(eval_uidxs)
        eval_uidxs = eval_uidxs[:n_eval]

        all_p, all_r, all_f1, all_ndcg, all_map = [], [], [], [], []
        all_s, all_ss, all_div, all_nov = [], [], [], []

        for uidx in eval_uidxs:
            row = test_mat[uidx]
            # rel = conjunto de ÍNDICES de columna (no recipe IDs) con rating >= 4
            rel = set(row.indices[row.data >= 4])
            if not rel:
                continue
            seen = set(train_mat[uidx].indices)
            scores = user_factors[uidx] @ item_factors.T
            if penalty is not None:
                scores = scores - alpha * penalty
            scores[list(seen)] = -np.inf

            top_k_iidxs = np.argsort(scores)[::-1][:k]
            top_k_rids  = [idx2iid[i] for i in top_k_iidxs]

            # hits usa ÍNDICES vs ÍNDICES (no recipe IDs)
            hits = [1 if iidx in rel else 0 for iidx in top_k_iidxs]
            p  = sum(hits) / k
            r  = sum(hits) / len(rel)
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.

            dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1. / np.log2(i + 2) for i in range(min(len(rel), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.

            n_hits, ap = 0, 0.
            for i, h in enumerate(hits):
                if h:
                    n_hits += 1
                    ap += n_hits / (i + 1)
            map_val = ap / min(len(rel), k) if rel else 0.

            sel = [recipe_sellos_dict.get(rid, 0) for rid in top_k_rids]
            all_p.append(p);   all_r.append(r);   all_f1.append(f1)
            all_ndcg.append(ndcg); all_map.append(map_val)
            all_s.append(float(np.mean(sel)))
            all_ss.append(float(np.mean([1 if s == 0 else 0 for s in sel])))

            # Diversidad y Novedad (recomendaciones híbridas)
            hist_ids  = [idx2iid[i] for i in train_mat[uidx].indices]
            cluster_u = pp.cluster_dominante(hist_ids, cluster_map)
            hybrid    = pp.recomendar(top_k_rids, hist_ids, cluster_map,
                                      pool_saludable, cluster_u, k=k,
                                      pool_con_sello=pool_con_sello)
            nd = pp.evaluar(hybrid, hist_ids, cluster_map,
                            pool_saludable, total_clusters, cluster_u)
            all_div.append(nd['Diversidad'])
            all_nov.append(nd['Novedad'])

        return {
            'Precision@K':           float(np.mean(all_p)),
            'Recall@K':              float(np.mean(all_r)),
            'F1@K':                  float(np.mean(all_f1)),
            'nDCG@K':                float(np.mean(all_ndcg)),
            'MAP@K':                 float(np.mean(all_map)),
            'AUC ROC':               0.0,
            'S@K (Promedio Sellos)': float(np.mean(all_s)),
            'SS@K (% Sin Sellos)':   float(np.mean(all_ss)),
            'Diversidad':            float(np.mean(all_div)),
            'Novedad':               float(np.mean(all_nov)),
        }

    print(f'Entrenando TruncatedSVD ({NO_COMPONENTS} componentes)...')
    t0 = time.time()
    svd = TruncatedSVD(n_components=NO_COMPONENTS, random_state=SEED, n_iter=10)
    uf  = svd.fit_transform(train_mat)
    if_ = svd.components_.T
    print(f'Entrenamiento: {time.time()-t0:.1f}s')

    print('Evaluando modelo base (sin sellos)...')
    res_sin = svd_evaluate(uf, if_)
    p_sin, r_sin, f1_sin = res_sin['Precision@K'], res_sin['Recall@K'], res_sin['F1@K']
    ndcg_sin, map_sin, auc_sin = res_sin['nDCG@K'], res_sin['MAP@K'], res_sin['AUC ROC']
    s_sin, ss_sin = res_sin['S@K (Promedio Sellos)'], res_sin['SS@K (% Sin Sellos)']
    div_sin, nov_sin = res_sin['Diversidad'], res_sin['Novedad']

    print('Evaluando modelo híbrido (con penalización sellos)...')
    res_con = svd_evaluate(uf, if_, penalty=health_penalty, alpha=ALPHA_HEALTH)
    p_con, r_con, f1_con = res_con['Precision@K'], res_con['Recall@K'], res_con['F1@K']
    ndcg_con, map_con, auc_con = res_con['nDCG@K'], res_con['MAP@K'], res_con['AUC ROC']
    s_con, ss_con = res_con['S@K (Promedio Sellos)'], res_con['SS@K (% Sin Sellos)']
    div_con, nov_con = res_con['Diversidad'], res_con['Novedad']

    # Guardar para ejemplo posterior
    _svd_uf = uf; _svd_if = if_; _svd_idx2iid = idx2iid; _svd_uid2idx = uid2idx
    _svd_train_mat = train_mat

# ── 5. Tabla comparativa ──────────────────────────────────────────────────────
backend = 'LightFM BPR' if USE_LIGHTFM else f'TruncatedSVD (k={NO_COMPONENTS})'

# Variables de diversidad/novedad existen solo en rama SVD; usa 0 en LightFM
if not USE_LIGHTFM:
    pass   # ya asignadas: div_sin, nov_sin, div_con, nov_con
else:
    div_sin = nov_sin = div_con = nov_con = 0.0

resultados = pd.DataFrame({
    'Métrica': ['Precision@K', 'Recall@K', 'F1@K', 'MAP@K', 'NDCG@K', 'AUC ROC',
                'S@K (Promedio Sellos)', 'SS@K (% Sin Sellos)',
                'Diversidad', 'Novedad'],
    'Sin Sellos (Base)': [
        p_sin, r_sin, f1_sin, map_sin, ndcg_sin, auc_sin, s_sin, ss_sin,
        div_sin, nov_sin,
    ],
    'Con Sellos (Híbrido)': [
        p_con, r_con, f1_con, map_con, ndcg_con, auc_con, s_con, ss_con,
        div_con, nov_con,
    ],
})
print(f'\n===== COMPARATIVA LightFM [{backend}] K={K} =====')
print(resultados.to_string(index=False))

resultados.to_csv(os.path.join(OUT_DIR, 'lfm_comparacion.csv'), index=False)
print('Guardado: lfm_comparacion.csv')

# ── 6. Gráfico barras agrupadas ───────────────────────────────────────────────
metrics_plot  = ['Precision@K', 'Recall@K', 'F1@K', 'NDCG@K']
vals_sin  = [p_sin,  r_sin,  f1_sin,  ndcg_sin]
vals_con  = [p_con,  r_con,  f1_con,  ndcg_con]
x     = np.arange(len(metrics_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width/2, vals_sin, width, label='Sin Sellos (Base)',    color='#2196F3', edgecolor='white')
b2 = ax.bar(x + width/2, vals_con, width, label='Con Sellos (Híbrido)', color='#F44336', edgecolor='white')
for bar in [*b1, *b2]:
    h = bar.get_height()
    ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'LightFM [{backend}] — Métricas de Relevancia (K={K})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(vals_sin + vals_con) * 1.4 if max(vals_sin + vals_con) > 0 else 0.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lfm_metricas.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: lfm_metricas.png')

# ── 7. Gráfico salud ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cats = ['S@K\n(Base)', 'S@K\n(Híbrido)', 'SS@K\n(Base)', 'SS@K\n(Híbrido)']
vals = [s_sin, s_con, ss_sin, ss_con]
cols = ['#F44336', '#F44336', '#9C27B0', '#9C27B0']
alphas = [0.9, 0.5, 0.9, 0.5]
bars = ax.bar(cats, vals, color=cols, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, vals):
    ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11)
ax.set_title(f'LightFM [{backend}] — Métricas de Salud (K={K})', fontsize=12)
ax.set_ylabel('Valor', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(vals) * 1.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lfm_salud.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: lfm_salud.png')

# ── 8. Gráfico Diversidad y Novedad ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels_dn = ['Diversidad\n(Base)', 'Diversidad\n(Híbrido)', 'Novedad\n(Base)', 'Novedad\n(Híbrido)']
vals_dn   = [div_sin, div_con, nov_sin, nov_con]
cols_dn   = ['#009688', '#00695C', '#FF9800', '#E65100']
bars = ax.bar(labels_dn, vals_dn, color=cols_dn, alpha=0.88, edgecolor='white')
for bar, val in zip(bars, vals_dn):
    ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11)
ax.set_title(f'LightFM [{backend}] — Diversidad y Novedad (K={K})', fontsize=12)
ax.set_ylabel('Valor', fontsize=12)
ax.set_ylim(0, max(vals_dn) * 1.4 if max(vals_dn) > 0 else 0.01)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lfm_diversidad_novedad.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: lfm_diversidad_novedad.png')

# ── 9. Ejemplo: 10 usuarios aleatorios (solo rama SVD) ───────────────────────
if not USE_LIGHTFM:
    print('\n' + '='*70)
    print('EJEMPLO: 10 usuarios aleatorios — recomendaciones híbridas')
    print('='*70)

    recipe_name_map = dict(zip(
        recipes_f['RecipeId'],
        recipes_f['Name'] if 'Name' in recipes_f.columns else recipes_f['RecipeId']
    ))

    random.seed(42)
    all_uids  = list(_svd_uid2idx.keys())
    sample_us = random.sample(all_uids, min(10, len(all_uids)))

    for uid in sample_us:
        uidx   = _svd_uid2idx[uid]
        seen   = set(_svd_train_mat[uidx].indices)
        hist_rids = [_svd_idx2iid[i] for i in seen]
        scores = _svd_uf[uidx] @ _svd_if.T
        scores[list(seen)] = -np.inf
        top_k_iidxs = np.argsort(scores)[::-1][:K]
        base_recs   = [_svd_idx2iid[i] for i in top_k_iidxs]
        cluster_u   = pp.cluster_dominante(hist_rids, cluster_map)
        hybrid      = pp.recomendar(base_recs, hist_rids, cluster_map, pool_saludable, cluster_u, k=K,
                                    pool_con_sello=pool_con_sello)
        nd          = pp.evaluar(hybrid, hist_rids, cluster_map, pool_saludable, total_clusters, cluster_u)
        print(f'\nUsuario {uid} | Cluster: {cluster_u} | '
              f'Diversidad: {nd["Diversidad"]:.2f} | Novedad: {nd["Novedad"]:.2f}')
        for j, rid in enumerate(hybrid, 1):
            nm  = recipe_name_map.get(rid, rid)
            cat = cluster_map.get(rid, '?')
            sel = recipe_sellos_dict.get(rid, 0)
            mark = '[saludable]' if sel == 0 else f'[{int(sel)} sello(s)]'
            print(f'  {j:2d}. {str(nm)[:45]:45s} | {cat[:25]:25s} | {mark}')

# ── Gráficos adicionales: Venn, Pareto, Clusters ──────────────────────────────
print('\nGenerando gráficos adicionales (Venn, Pareto, Clusters)...')

try:
    from venn import venn as _venn_fn
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, '-m', 'pip', 'install', 'venn', '-q'])
    from venn import venn as _venn_fn

# Recolectar recetas (SVD base: sin penalización de salud)
if not USE_LIGHTFM:
    random.seed(42)
    _lfm_uids = list(_svd_uid2idx.keys())
    random.shuffle(_lfm_uids)
    _lfm_rids = []
    for _uid in _lfm_uids[:500]:
        _ux   = _svd_uid2idx[_uid]
        _seen = set(_svd_train_mat[_ux].indices)
        _scr  = _svd_uf[_ux] @ _svd_if.T
        _scr[list(_seen)] = -np.inf
        _lfm_rids.extend([_svd_idx2iid[i] for i in np.argsort(_scr)[::-1][:K]])
else:
    _lfm_rids = []

if _lfm_rids:
    # Venn
    _lfm_rf = recipes_f[recipes_f['RecipeId'].isin(set(_lfm_rids))]
    _hcols  = ['IsHighSaturatedFat', 'IsHighSugar', 'IsHighSodium', 'IsHighCalories']
    _hnames = ['Grasas Saturadas', 'Azúcares', 'Sodio', 'Calorías']
    _vsets  = {n: set(_lfm_rf[_lfm_rf[c] == True]['RecipeId']) for c, n in zip(_hcols, _hnames)}
    plt.figure(figsize=(10, 10))
    _venn_fn(_vsets)
    plt.title(f'LightFM [{backend}] — Intersección de Sellos en Recetas Recomendadas', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'lfm_venn_sellos.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Guardado: lfm_venn_sellos.png')

    # Clusters
    _lfm_cats = pd.Series([cluster_map.get(rid, 'Unknown') for rid in _lfm_rids]).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    _lfm_cats.plot(kind='bar', ax=ax, color='mediumseagreen', edgecolor='white')
    ax.set_title(f'LightFM [{backend}] — Top-20 Clusters en Recetas Recomendadas', fontsize=13)
    ax.set_xlabel('Categoría (Cluster)', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'lfm_clusters.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Guardado: lfm_clusters.png')

# Pareto: dos puntos — base vs híbrido
_lfm_pts = [
    ('Base\n(sin sellos)',    ndcg_sin, ss_sin),
    ('Híbrido\n(con sellos)', ndcg_con, ss_con),
]
fig, ax = plt.subplots(figsize=(7, 6))
_lfm_cols = ['#1976D2', '#E53935']
for (_lbl, _nd, _ss), _col in zip(_lfm_pts, _lfm_cols):
    ax.scatter([_nd], [_ss], s=200, color=_col, zorder=5, edgecolors='black', lw=1.2)
    ax.annotate(_lbl, xy=(_nd, _ss), xytext=(8, 4), textcoords='offset points', fontsize=10)
# Conectar los dos puntos con una línea
ax.plot([ndcg_sin, ndcg_con], [ss_sin, ss_con], '--', color='gray', lw=1.5, alpha=0.7)
ax.set_xlabel('nDCG@10 ↑  (Relevancia)', fontsize=12)
ax.set_ylabel('SS@10 ↑  (% sin sellos)', fontsize=12)
ax.set_title(f'LightFM [{backend}] — Trade-off Relevancia vs Salud', fontsize=12)
ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lfm_pareto.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: lfm_pareto.png')

print('\n¡Listo!')
