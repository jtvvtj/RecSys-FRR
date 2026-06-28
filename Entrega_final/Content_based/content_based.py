"""
Content-Based Filtering con TF-IDF + TruncatedSVD
Replica de Content_based_filtrado.ipynb (Entrega Intermedia)
Salida: Tablas_graficos/
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from post_processor import RecommenderPostProcessor
pp = RecommenderPostProcessor()

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'Tablas_graficos')
os.makedirs(OUT_DIR, exist_ok=True)

def _find_file(filename, extra_dirs=()):
    """Busca un archivo en ubicaciones conocidas."""
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

# ── Parámetros ───────────────────────────────────────────────────────────────
K             = 10
RANDOM_STATE  = 42
REL_THRESHOLD = 4   # rating >= 4 es relevante

# ── Carga ────────────────────────────────────────────────────────────────────
print('Cargando datos...')
reviews = pd.read_csv(REVIEWS_PATH)
recipes = pd.read_csv(RECIPES_PATH)
recipes['ExtractedServingSize'] = (
    recipes['ExtractedServingSize'].str.extract(r'\((.*?)\)').astype(float)
)

# ── Filtrado ─────────────────────────────────────────────────────────────────
review_counts        = reviews['AuthorId'].value_counts()
single_review_users  = review_counts[review_counts == 1].index
reviews_f = reviews[~reviews['AuthorId'].isin(single_review_users)].copy()
recipes_f = recipes[recipes['RecipeId'].isin(reviews_f['RecipeId'].unique())].copy()
recipes_f = recipes_f[recipes_f['ExtractedServingSize'] > 0].copy()
reviews_f = reviews_f[reviews_f['RecipeId'].isin(recipes_f['RecipeId'].unique())].copy()

print(f'Usuarios: {reviews_f["AuthorId"].nunique()} | '
      f'Recetas: {recipes_f["RecipeId"].nunique()} | '
      f'Ratings: {len(reviews_f)}')

# ── Corrección serving size (batches multi-unidad con ConsolidatedServings=1) ─
from post_processor import fix_serving_sizes as _fss
recipes_f = _fss(recipes_f)

# ── Sellos Ley 20.606 (por 100 g) ───────────────────────────────────────────
sv = recipes_f['ExtractedServingSize']
recipes_f = recipes_f.copy()
recipes_f['IsHighCalories']    = (recipes_f['Calories']           / sv * 100) >= 275
recipes_f['IsHighSugar']       = (recipes_f['SugarContent']       / sv * 100) >= 10
recipes_f['IsHighSaturatedFat']= (recipes_f['SaturatedFatContent']/ sv * 100) >= 4
recipes_f['IsHighSodium']      = (recipes_f['SodiumContent']      / sv * 100) >= 400
recipes_f['num_sellos']        = recipes_f[
    ['IsHighCalories','IsHighSugar','IsHighSaturatedFat','IsHighSodium']
].sum(axis=1).astype(int)
recipe_sellos_dict = dict(zip(recipes_f['RecipeId'], recipes_f['num_sellos']))

# ── Cluster map y pool saludable ─────────────────────────────────────────────
cluster_map    = dict(zip(recipes_f['RecipeId'], recipes_f['RecipeCategory'].fillna('Unknown')))
pool_saludable = set(recipes_f.loc[recipes_f['num_sellos'] == 0, 'RecipeId'])
pool_con_sello = set(cluster_map.keys()) - pool_saludable
total_clusters = recipes_f['RecipeCategory'].nunique()
print(f'Clusters (categorías): {total_clusters} | Pool saludable: {len(pool_saludable):,}')

# ── TF-IDF + TruncatedSVD ────────────────────────────────────────────────────
print('Construyendo embedding TF-IDF + SVD (50 componentes)...')
recipes_f = recipes_f.copy()
recipes_f['text'] = (
    recipes_f['Name'].fillna('') + ' ' + recipes_f['Description'].fillna('')
)
vectorizer   = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_sparse = vectorizer.fit_transform(recipes_f['text'])
tfidf_mat    = TruncatedSVD(n_components=50, random_state=RANDOM_STATE).fit_transform(tfidf_sparse)

idx2rid  = {i: rid for i, rid in enumerate(recipes_f['RecipeId'])}
rid2idx  = {rid: i for i, rid in idx2rid.items()}

# Normalizar para similitud coseno
norms  = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
norms[norms == 0] = 1
emb    = tfidf_mat / norms   # shape (n_recipes, 50)

# ── Split 80/20 ──────────────────────────────────────────────────────────────
print('Split 80/20...')
train_df, test_df = train_test_split(
    reviews_f, test_size=0.2, random_state=RANDOM_STATE
)

user_train_items: dict[int, list] = {}
for uid, grp in train_df.groupby('AuthorId'):
    user_train_items[uid] = grp['RecipeId'].tolist()

# ── Funciones de métricas ─────────────────────────────────────────────────────
def _dcg(rel, k):
    r = np.asarray(rel, dtype=float)[:k]
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2))) if r.size else 0.

def ndcg_at_k(rec, relevant, k=K):
    rel = [1 if r in relevant else 0 for r in rec[:k]]
    dcg  = _dcg(rel, k)
    idcg = _dcg(sorted(rel, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.

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
    return float(np.mean([1 if recipe_sellos_dict.get(r, 0) == 0 else 0
                           for r in rec[:k]]))

# ── Recomendación vectorizada ─────────────────────────────────────────────────
def recommend_fast(uid: int, topk: int = K):
    trx = user_train_items.get(uid, [])
    hidx = [rid2idx[t] for t in trx if t in rid2idx]
    if not hidx:
        return []
    hvecs   = emb[hidx]                        # (n_hist, 50)
    sim_all = (hvecs @ emb.T).max(axis=0)      # (n_recipes,)
    for i in hidx:
        sim_all[i] = -1.0
    top_idx = np.argpartition(sim_all, -topk)[-topk:]
    top_idx = top_idx[np.argsort(sim_all[top_idx])[::-1]]
    trx_set = set(trx)
    return [idx2rid[i] for i in top_idx if idx2rid[i] not in trx_set][:topk]

# ── Evaluación ────────────────────────────────────────────────────────────────
test_user_ids = test_df['AuthorId'].unique()
print(f'Evaluando {len(test_user_ids)} usuarios...')
start = time.time()

# Índice test: user -> relevant set
test_relevant: dict[int, set] = {}
for uid, grp in test_df.groupby('AuthorId'):
    test_relevant[uid] = set(grp.loc[grp['Rating'] >= REL_THRESHOLD, 'RecipeId'])

all_metrics = []
for i, uid in enumerate(test_user_ids):
    recs = recommend_fast(uid)
    if not recs:
        continue
    relevant = test_relevant.get(uid, set())
    hits = len(set(recs) & relevant)
    p  = hits / K
    r  = hits / len(relevant) if relevant else 0.
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.
    hist        = user_train_items.get(uid, [])
    cluster_u   = pp.cluster_dominante(hist, cluster_map)
    hybrid_recs = pp.recomendar(recs, hist, cluster_map, pool_saludable, cluster_u, k=K,
                               pool_con_sello=pool_con_sello)
    nd          = pp.evaluar(hybrid_recs, hist, cluster_map, pool_saludable, total_clusters, cluster_u)
    all_metrics.append({
        'P@K':       p,
        'R@K':       r,
        'F1@K':      f1,
        'nDCG@K':    ndcg_at_k(recs, relevant),
        'MAP@K':     map_at_k(recs, relevant),
        'S@K':       sellos_at_k(recs),
        'SS@K':      sello_free_at_k(recs),
        'Diversidad': nd['Diversidad'],
        'Novedad':    nd['Novedad'],
    })
    if (i + 1) % 5000 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        print(f'  {i+1}/{len(test_user_ids)} ({rate:.1f} users/s, '
              f'~{(len(test_user_ids)-i-1)/rate/60:.0f} min restantes)')

results = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
elapsed = time.time() - start

print(f'\n=== Content-Based (TF-IDF + SVD) K={K} ===')
print(f'Usuarios evaluados: {len(all_metrics)}')
for metric, val in results.items():
    print(f'  {metric}: {val:.6f}')
print(f'  Tiempo total: {elapsed/60:.1f} min')

# ── Guardar tabla ─────────────────────────────────────────────────────────────
df_out = pd.DataFrame([results])
df_out.insert(0, 'Modelo', 'Content-Based (TF-IDF + SVD)')
df_out.to_csv(os.path.join(OUT_DIR, 'cb_resultados.csv'), index=False)
print(f'\nTabla guardada: Tablas_graficos/cb_resultados.csv')

# ── Gráfico de relevancia ──────────────────────────────────────────────────────
labels_rel = ['P@10', 'R@10', 'F1@10', 'nDCG@10']
keys_rel   = ['P@K',  'R@K',  'F1@K',  'nDCG@K']
vals_rel   = [results[k] for k in keys_rel]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels_rel, vals_rel, color='#4472C4', edgecolor='white', width=0.5)
ax.set_xlabel('Métrica', fontsize=12)
ax.set_ylabel('Score',   fontsize=12)
ax.set_title('Content-Based (TF-IDF + SVD) — Métricas de Relevancia (K=10)', fontsize=13)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(vals_rel) * 1.4 if max(vals_rel) > 0 else 0.005)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_relevancia.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: Tablas_graficos/cb_relevancia.png')

# ── Gráfico de salud ───────────────────────────────────────────────────────────
h_labels = ['S@10\n(avg. sellos  ↓ mejor)', 'SS@10\n(% libre de sellos  ↑ mejor)']
h_vals   = [results['S@K'], results['SS@K']]
h_colors = ['#F44336', '#9C27B0']

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(h_labels, h_vals, width=0.4, color=h_colors, edgecolor='white')
fmt_labels = [f'{h_vals[0]:.3f}', f'{h_vals[1]*100:.1f} %']
for bar, lbl in zip(bars, fmt_labels):
    ax.annotate(lbl,
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=14, fontweight='bold')
ax.set_title('Content-Based (TF-IDF + SVD) — Métricas de Salud\n(K=10)', fontsize=13)
ax.set_ylabel('Valor', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(h_vals) * 1.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_salud.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: Tablas_graficos/cb_salud.png')
# ── Gráfico Diversidad y Novedad ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
dn_labels = ['Diversidad', 'Novedad']
dn_vals   = [results['Diversidad'], results['Novedad']]
dn_colors = ['#1976D2', '#43A047']
bars = ax.bar(dn_labels, dn_vals, width=0.4, color=dn_colors, edgecolor='white')
for bar, val in zip(bars, dn_vals):
    ax.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=13, fontweight='bold')
ax.set_title('Content-Based — Diversidad y Novedad (K=10)', fontsize=13)
ax.set_ylabel('Score (0–1)', fontsize=12)
ax.set_ylim(0, max(dn_vals) * 1.4 if max(dn_vals) > 0 else 0.1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_diversidad_novedad.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: Tablas_graficos/cb_diversidad_novedad.png')

# ── Ejemplo 10 usuarios aleatorios ────────────────────────────────────────────
print('\n=== Ejemplo 10 usuarios aleatorios (Recomendación Híbrida) ===')
sample_uids = random.sample(list(user_train_items.keys()), min(10, len(user_train_items)))
recipe_name_dict = dict(zip(recipes_f['RecipeId'], recipes_f['Name']))
for uid in sample_uids:
    recs_base = recommend_fast(uid)
    if not recs_base:
        continue
    hist      = user_train_items.get(uid, [])
    cluster_u = pp.cluster_dominante(hist, cluster_map)
    hybrid    = pp.recomendar(recs_base, hist, cluster_map, pool_saludable, cluster_u, k=K,
                             pool_con_sello=pool_con_sello)
    nd        = pp.evaluar(hybrid, hist, cluster_map, pool_saludable, total_clusters, cluster_u)
    print(f'\nUsuario {uid} | cluster={cluster_u} | '
          f'Div={nd["Diversidad"]:.3f} | Nov={nd["Novedad"]:.3f}')
    for rank, rid in enumerate(hybrid, 1):
        s = recipe_sellos_dict.get(rid, '?')
        print(f'  {rank:2}. {recipe_name_dict.get(rid, rid):<45} [{s} sellos]')

# ── Gráficos adicionales: Venn, Pareto, Clusters ──────────────────────────────
print('\nGenerando gráficos adicionales (Venn, Pareto, Clusters)...')

try:
    from venn import venn as _venn_fn
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, '-m', 'pip', 'install', 'venn', '-q'])
    from venn import venn as _venn_fn

# Recolectar recetas recomendadas de 2000 usuarios al azar
random.seed(42)
_cb_uids = random.sample(list(user_train_items.keys()), min(2000, len(user_train_items)))
_cb_rids = []
for _uid in _cb_uids:
    _cb_rids.extend(recommend_fast(_uid))

# Venn: intersección de sellos en recetas recomendadas
_cb_rf = recipes_f[recipes_f['RecipeId'].isin(set(_cb_rids))]
_hcols  = ['IsHighSaturatedFat', 'IsHighSugar', 'IsHighSodium', 'IsHighCalories']
_hnames = ['Grasas Saturadas', 'Azúcares', 'Sodio', 'Calorías']
_venn_sets = {n: set(_cb_rf[_cb_rf[c] == True]['RecipeId']) for c, n in zip(_hcols, _hnames)}
plt.figure(figsize=(10, 10))
_venn_fn(_venn_sets)
plt.title('Content-Based — Intersección de Sellos en Recetas Recomendadas', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_venn_sellos.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: cb_venn_sellos.png')

# Pareto: nDCG@10 vs SS@10 (único punto para CB)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter([results['nDCG@K']], [results['SS@K']], s=220, color='#1976D2',
           zorder=5, edgecolors='black', linewidths=1.3)
ax.annotate('Content-Based\n(K=10)', xy=(results['nDCG@K'], results['SS@K']),
            xytext=(10, 8), textcoords='offset points', fontsize=10)
ax.set_xlabel('nDCG@10 ↑  (Relevancia)', fontsize=12)
ax.set_ylabel('SS@10 ↑  (% sin sellos)', fontsize=12)
ax.set_title('Content-Based — Trade-off Relevancia vs Salud', fontsize=13)
ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_pareto.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: cb_pareto.png')

# Clusters: distribución de categorías en recetas recomendadas
_cb_cats = pd.Series([cluster_map.get(rid, 'Unknown') for rid in _cb_rids]).value_counts().head(20)
fig, ax = plt.subplots(figsize=(14, 6))
_cb_cats.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Content-Based — Top-20 Clusters en Recetas Recomendadas', fontsize=13)
ax.set_xlabel('Categoría (Cluster)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cb_clusters.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: cb_clusters.png')

print('\n¡Listo!')
