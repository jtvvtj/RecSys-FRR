"""
Q-Learning para Recomendación de Recetas con Sellos Nutricionales
Replica de Q_Learning_RecSys.ipynb (Entrega Intermedia)
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
from collections import defaultdict

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

# ── Parámetros ───────────────────────────────────────────────────────────────
NUM_ACTIONS  = 500
NUM_EPOCHS   = 5
K            = 10

# ── 1. Carga y filtrado ───────────────────────────────────────────────────────
print('Cargando datos...')
reviews = pd.read_csv(REVIEWS_PATH)
recipes = pd.read_csv(RECIPES_PATH)
recipes['ExtractedServingSize'] = (
    recipes['ExtractedServingSize'].str.extract(r'\((.*?)\)').astype(float)
)

review_counts       = reviews['AuthorId'].value_counts()
single_review_users = review_counts[review_counts == 1].index
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

# ── 2. Sellos Ley 20.606 (por 100 g) ─────────────────────────────────────────
sv = recipes_f['ExtractedServingSize']
recipes_f = recipes_f.copy()
recipes_f['IsHighCalories']     = (recipes_f['Calories']            / sv * 100) >= 275
recipes_f['IsHighSugar']        = (recipes_f['SugarContent']        / sv * 100) >= 10
recipes_f['IsHighSaturatedFat'] = (recipes_f['SaturatedFatContent'] / sv * 100) >= 4
recipes_f['IsHighSodium']       = (recipes_f['SodiumContent']       / sv * 100) >= 400
recipes_f['num_sellos']         = recipes_f[
    ['IsHighCalories', 'IsHighSugar', 'IsHighSaturatedFat', 'IsHighSodium']
].sum(axis=1)
recipe_sellos  = recipes_f.set_index('RecipeId')['num_sellos'].to_dict()
cluster_map    = dict(zip(recipes_f['RecipeId'], recipes_f['RecipeCategory'].fillna('Unknown')))
pool_saludable = list(recipes_f.loc[recipes_f['num_sellos'] == 0, 'RecipeId'])
pool_con_sello = set(cluster_map.keys()) - set(pool_saludable)
total_clusters = recipes_f['RecipeCategory'].nunique()

# ── 3. Espacio de acciones: top-500 recetas más populares ─────────────────────
recipe_popularity = reviews_f.groupby('RecipeId')['Rating'].count().sort_values(ascending=False)
top_recipes       = recipe_popularity.head(NUM_ACTIONS).index.tolist()
recipe_to_action  = {rid: i for i, rid in enumerate(top_recipes)}
action_to_recipe  = {i: rid for rid, i in recipe_to_action.items()}

print(f'Espacio de acciones: {NUM_ACTIONS} recetas')
print(f'Promedio de sellos en top-{NUM_ACTIONS}: '
      f'{np.mean([recipe_sellos.get(r, 0) for r in top_recipes]):.2f}')

# ── 4. Split temporal (solo reviews del espacio de acciones) ──────────────────
reviews_sorted  = reviews_f.sort_values('DateSubmitted')
reviews_actions = reviews_sorted[reviews_sorted['RecipeId'].isin(top_recipes)].copy()

user_interactions_train: dict = {}
user_interactions_test:  dict = {}
for user_id, group in reviews_actions.groupby('AuthorId'):
    items = list(zip(group['RecipeId'].tolist(), group['Rating'].tolist()))
    if len(items) < 2:
        continue
    user_interactions_train[user_id] = items[:-1]
    user_interactions_test[user_id]  = items[-1:]

print(f'Usuarios train/test: {len(user_interactions_train)} | '
      f'Interacciones train: {sum(len(v) for v in user_interactions_train.values())}')

# ── 5. Estado y recompensa ────────────────────────────────────────────────────
def discretize_state(avg_rating, num_interactions, avg_sellos):
    r_level = 0 if avg_rating < 2 else (1 if avg_rating < 4 else 2)
    a_level = 0 if num_interactions <= 3 else (1 if num_interactions <= 10 else 2)
    s_level = 0 if avg_sellos < 1 else (1 if avg_sellos < 2 else 2)
    return (r_level, a_level, s_level)

def get_user_state(interactions, recipe_sellos_dict):
    if not interactions:
        return (1, 0, 1)
    ratings = [r for _, r in interactions]
    sellos  = [recipe_sellos_dict.get(rid, 0) for rid, _ in interactions]
    return discretize_state(np.mean(ratings), len(interactions), np.mean(sellos))

def compute_reward(rating, recipe_id, recipe_sellos_dict, alpha=0.0):
    relevance     = 1.0 if rating >= 4 else (0.0 if rating == 3 else -1.0)
    health_penalty = recipe_sellos_dict.get(recipe_id, 0) / 4.0
    return relevance - alpha * health_penalty

# ── 6. Agente Q-Learning ──────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions   = num_actions
        self.lr            = learning_rate
        self.gamma         = discount_factor
        self.epsilon       = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = defaultdict(lambda: np.zeros(num_actions))

    def update(self, state, action, reward, next_state):
        best_next  = np.max(self.q_table[next_state])
        td_error   = reward + self.gamma * best_next - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_top_k(self, state, k=10, exclude_actions=None):
        q_values = self.q_table[state].copy()
        if exclude_actions:
            for a in exclude_actions:
                q_values[a] = -np.inf
        return np.argsort(q_values)[::-1][:k]

# ── 7. Entrenamiento ──────────────────────────────────────────────────────────
def train_agent(agent, interactions_train, recipe_to_action, recipe_sellos,
                alpha=0.0, num_epochs=NUM_EPOCHS):
    rewards_history = []
    users = list(interactions_train.keys())
    for epoch in range(num_epochs):
        epoch_rewards = []
        np.random.shuffle(users)
        for user_id in users:
            history = []
            for recipe_id, rating in interactions_train[user_id]:
                if recipe_id not in recipe_to_action:
                    continue
                state  = get_user_state(history, recipe_sellos)
                action = recipe_to_action[recipe_id]
                reward = compute_reward(rating, recipe_id, recipe_sellos, alpha=alpha)
                history.append((recipe_id, rating))
                next_state = get_user_state(history, recipe_sellos)
                agent.update(state, action, reward, next_state)
                epoch_rewards.append(reward)
        agent.decay_epsilon()
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        rewards_history.append(avg_reward)
        print(f'  Epoch {epoch+1}/{num_epochs} - '
              f'Avg Reward: {avg_reward:.4f} - Epsilon: {agent.epsilon:.4f}')
    return rewards_history

# ── 8. Métricas ───────────────────────────────────────────────────────────────
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
    return float(np.mean([recipe_sellos.get(r, 0) for r in rec[:k]]))

def sello_free_at_k(rec, k=K):
    return float(np.mean([1 if recipe_sellos.get(r, 0) == 0 else 0 for r in rec[:k]]))

def evaluate_agent(agent, interactions_train, interactions_test,
                   recipe_to_action, action_to_recipe, recipe_sellos, k=K):
    metrics = {'precision': [], 'recall': [], 'ndcg': [], 'map': [],
               's_at_k': [], 'ss_at_k': [], 'diversidad': [], 'novedad': []}
    test_users = list(interactions_test.keys())
    for i, user_id in enumerate(test_users):
        if (i + 1) % 2000 == 0:
            print(f'    {i+1}/{len(test_users)}', end='\r')
        train_hist = interactions_train.get(user_id, [])
        test_items = interactions_test[user_id]
        state      = get_user_state(train_hist, recipe_sellos)
        seen_actions = {recipe_to_action[rid] for rid, _ in train_hist if rid in recipe_to_action}
        top_k_actions  = agent.get_top_k(state, k=k, exclude_actions=seen_actions)
        rec = [action_to_recipe[a] for a in top_k_actions]
        relevant = {rid for rid, rating in test_items if rating >= 4}
        hits = len(set(rec) & relevant)
        p  = hits / k
        r  = hits / len(relevant) if relevant else 0.
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['ndcg'].append(ndcg_at_k(rec, relevant, k))
        metrics['map'].append(map_at_k(rec, relevant, k))
        metrics['s_at_k'].append(sellos_at_k(rec, k))
        metrics['ss_at_k'].append(sello_free_at_k(rec, k))
        # Diversidad y Novedad con recomendaciones híbridas
        hist_ids  = [rid for rid, _ in train_hist]
        cluster_u = pp.cluster_dominante(hist_ids, cluster_map)
        hybrid    = pp.recomendar(rec, hist_ids, cluster_map, pool_saludable, cluster_u, k=k,
                               pool_con_sello=pool_con_sello)
        nd        = pp.evaluar(hybrid, hist_ids, cluster_map, pool_saludable, total_clusters, cluster_u)
        metrics['diversidad'].append(nd['Diversidad'])
        metrics['novedad'].append(nd['Novedad'])
    print()
    return {m: float(np.mean(v)) for m, v in metrics.items()}

# ── 9. Entrenar agentes sin y con sellos ─────────────────────────────────────
print('\n' + '='*60)
print('Entrenando Q-Learning SIN sellos (alpha=0)')
print('='*60)
agent_no  = QLearningAgent(num_actions=NUM_ACTIONS)
t0 = time.time()
rewards_no = train_agent(agent_no, user_interactions_train,
                          recipe_to_action, recipe_sellos, alpha=0.0)
print(f'Tiempo: {time.time()-t0:.1f}s')

print('\n' + '='*60)
print('Entrenando Q-Learning CON sellos (alpha=0.5)')
print('='*60)
agent_con = QLearningAgent(num_actions=NUM_ACTIONS)
t0 = time.time()
rewards_con = train_agent(agent_con, user_interactions_train,
                           recipe_to_action, recipe_sellos, alpha=0.5)
print(f'Tiempo: {time.time()-t0:.1f}s')

# ── 10. Curva de aprendizaje ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, NUM_EPOCHS+1), rewards_no,  'b-o', label='Sin sellos (α=0)')
ax.plot(range(1, NUM_EPOCHS+1), rewards_con, 'r-o', label='Con sellos (α=0.5)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Recompensa Promedio')
ax.set_title('Curva de Aprendizaje Q-Learning')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_curva_aprendizaje.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: ql_curva_aprendizaje.png')

# ── 11. Evaluación de ambos agentes ──────────────────────────────────────────
print('\nEvaluando Q-Learning SIN sellos...')
res_no  = evaluate_agent(agent_no,  user_interactions_train, user_interactions_test,
                          recipe_to_action, action_to_recipe, recipe_sellos)
print('Evaluando Q-Learning CON sellos (α=0.5)...')
res_con = evaluate_agent(agent_con, user_interactions_train, user_interactions_test,
                          recipe_to_action, action_to_recipe, recipe_sellos)

results_df = pd.DataFrame({
    'Métrica': [f'Precision@{K}', f'Recall@{K}', f'nDCG@{K}', 'MAP',
                f'S@{K} (avg sellos)', f'SS@{K} (% sin sellos)',
                'Diversidad', 'Novedad'],
    'Q-Learning (sin sellos)': [
        round(res_no['precision'],  4), round(res_no['recall'],    4),
        round(res_no['ndcg'],       4), round(res_no['map'],       4),
        round(res_no['s_at_k'],     4), round(res_no['ss_at_k'],  4),
        round(res_no['diversidad'], 4), round(res_no['novedad'],  4),
    ],
    'Q-Learning (con sellos, α=0.5)': [
        round(res_con['precision'],  4), round(res_con['recall'],    4),
        round(res_con['ndcg'],       4), round(res_con['map'],       4),
        round(res_con['s_at_k'],     4), round(res_con['ss_at_k'],  4),
        round(res_con['diversidad'], 4), round(res_con['novedad'],  4),
    ],
})
print('\n' + results_df.to_string(index=False))

# ── 12. Tabla comparativa ─────────────────────────────────────────────────────
comparison = pd.DataFrame({
    'Modelo': [
        'Content-Based (K=10)',
        f'Q-Learning sin sellos (α=0)',
        f'Q-Learning con sellos (α=0.5)',
    ],
    f'Precision@{K}': [0.002, round(res_no['precision'], 4), round(res_con['precision'], 4)],
    f'Recall@{K}':    [0.02,  round(res_no['recall'],    4), round(res_con['recall'],    4)],
    f'nDCG@{K}':      [0.02,  round(res_no['ndcg'],      4), round(res_con['ndcg'],      4)],
    f'S@{K}':         ['—',   round(res_no['s_at_k'],    4), round(res_con['s_at_k'],    4)],
    f'SS@{K}':        ['—',   round(res_no['ss_at_k'],   4), round(res_con['ss_at_k'],   4)],
    'Diversidad':     ['—',   round(res_no['diversidad'],4), round(res_con['diversidad'],4)],
    'Novedad':        ['—',   round(res_no['novedad'],   4), round(res_con['novedad'],   4)],
})
comparison.to_csv(os.path.join(OUT_DIR, 'ql_comparacion.csv'), index=False)
print('\nTabla comparativa guardada: ql_comparacion.csv')

# ── 13. Análisis de sensibilidad de α ────────────────────────────────────────
alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
sensitivity_results = []

for alpha in alphas:
    print(f'\n--- Entrenando con α={alpha} ---')
    agent = QLearningAgent(num_actions=NUM_ACTIONS)
    train_agent(agent, user_interactions_train, recipe_to_action, recipe_sellos,
                alpha=alpha, num_epochs=NUM_EPOCHS)
    res = evaluate_agent(agent, user_interactions_train, user_interactions_test,
                          recipe_to_action, action_to_recipe, recipe_sellos)
    res['alpha'] = alpha
    sensitivity_results.append(res)
    print(f'  precision={res["precision"]:.4f} | ndcg={res["ndcg"]:.4f} | '
          f's_at_k={res["s_at_k"]:.4f} | ss_at_k={res["ss_at_k"]:.4f}')

sensitivity_df = pd.DataFrame(sensitivity_results)
col_order = ['alpha', 'precision', 'recall', 'ndcg', 'map',
             's_at_k', 'ss_at_k', 'diversidad', 'novedad']
sensitivity_df = sensitivity_df[[c for c in col_order if c in sensitivity_df.columns]]
sensitivity_df.to_csv(os.path.join(OUT_DIR, 'ql_sensibilidad.csv'), index=False)
print('\n' + sensitivity_df.to_string(index=False))

# ── 14. Gráfico de sensibilidad (2x2) ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(sensitivity_df['alpha'], sensitivity_df['precision'], 'b-o')
axes[0, 0].set_xlabel('α (peso salud)')
axes[0, 0].set_ylabel(f'Precision@{K}')
axes[0, 0].set_title(f'Precision@{K} vs α')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(sensitivity_df['alpha'], sensitivity_df['ndcg'], 'g-o')
axes[0, 1].set_xlabel('α (peso salud)')
axes[0, 1].set_ylabel(f'nDCG@{K}')
axes[0, 1].set_title(f'nDCG@{K} vs α')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(sensitivity_df['alpha'], sensitivity_df['s_at_k'], 'r-o')
axes[1, 0].set_xlabel('α (peso salud)')
axes[1, 0].set_ylabel(f'S@{K} (avg sellos)')
axes[1, 0].set_title(f'S@{K} vs α (menor = más saludable)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(sensitivity_df['alpha'], sensitivity_df['ss_at_k'], 'm-o')
axes[1, 1].set_xlabel('α (peso salud)')
axes[1, 1].set_ylabel(f'SS@{K} (% sin sellos)')
axes[1, 1].set_title(f'SS@{K} vs α (mayor = más saludable)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Análisis de Sensibilidad: Trade-off Relevancia vs Salud', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_sensibilidad.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: ql_sensibilidad.png')

# ── 15. Gráfico trade-off (eje dual) ─────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('α (peso del factor de salud)')
ax1.set_ylabel(f'nDCG@{K}', color='tab:blue')
ax1.plot(sensitivity_df['alpha'], sensitivity_df['ndcg'], 'b-o', label=f'nDCG@{K}')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel(f'S@{K} (avg sellos)', color='tab:red')
ax2.plot(sensitivity_df['alpha'], sensitivity_df['s_at_k'], 'r-s', label=f'S@{K}')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.title('Trade-off: Relevancia (nDCG) vs Salud (Sellos)')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: ql_tradeoff.png')

# ── 16. Distribución de sellos en recomendaciones (500 usuarios) ──────────────
all_recs_no  = []
all_recs_con = []
for user_id in list(user_interactions_test.keys())[:500]:
    train_hist = user_interactions_train.get(user_id, [])
    st   = get_user_state(train_hist, recipe_sellos)
    seen = {recipe_to_action[rid] for rid, _ in train_hist if rid in recipe_to_action}
    for a in agent_no.get_top_k(st,  k=K, exclude_actions=seen):
        all_recs_no.append(recipe_sellos.get(action_to_recipe[a], 0))
    for a in agent_con.get_top_k(st, k=K, exclude_actions=seen):
        all_recs_con.append(recipe_sellos.get(action_to_recipe[a], 0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(all_recs_no,  bins=range(6), align='left', color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Número de sellos')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de sellos - Q-Learning SIN sellos')
axes[0].set_xticks(range(5))
axes[1].hist(all_recs_con, bins=range(6), align='left', color='coral',     edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Número de sellos')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de sellos - Q-Learning CON sellos (α=0.5)')
axes[1].set_xticks(range(5))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_distribucion_sellos.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Guardado: ql_distribucion_sellos.png')

print(f'\nPromedio sellos SIN factor salud: {np.mean(all_recs_no):.3f}')
print(f'Promedio sellos CON factor salud: {np.mean(all_recs_con):.3f}')

# ── 17. Gráfico Diversidad y Novedad vs α ─────────────────────────────────────
if 'diversidad' in sensitivity_df.columns and 'novedad' in sensitivity_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sensitivity_df['alpha'], sensitivity_df['diversidad'], 'teal', marker='o')
    axes[0].set_xlabel('α (peso salud)')
    axes[0].set_ylabel('Diversidad')
    axes[0].set_title(f'Diversidad@{K} vs α\n(clusters únicos / k)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sensitivity_df['alpha'], sensitivity_df['novedad'], 'darkorange', marker='s')
    axes[1].set_xlabel('α (peso salud)')
    axes[1].set_ylabel('Novedad')
    axes[1].set_title(f'Novedad@{K} vs α\n(% recs nuevas, saludables y del cluster)')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Diversidad y Novedad (recomendaciones híbridas) vs α', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ql_diversidad_novedad.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Guardado: ql_diversidad_novedad.png')

# ── 18. Ejemplo: 10 usuarios aleatorios con recomendaciones híbridas ──────────
print('\n' + '='*70)
print('EJEMPLO: 10 usuarios aleatorios — recomendaciones híbridas')
print('='*70)

recipe_name_map = dict(zip(recipes_f['RecipeId'],
                           recipes_f.get('Name', recipes_f.get('RecipeName', recipes_f['RecipeId']))))

random.seed(42)
sample_users = random.sample(list(user_interactions_test.keys()), min(10, len(user_interactions_test)))

for uid in sample_users:
    train_hist = user_interactions_train.get(uid, [])
    hist_ids   = [rid for rid, _ in train_hist]
    state      = get_user_state(train_hist, recipe_sellos)
    seen_acts  = {recipe_to_action[rid] for rid in hist_ids if rid in recipe_to_action}
    base_recs  = [action_to_recipe[a] for a in agent_con.get_top_k(state, k=K, exclude_actions=seen_acts)]
    cluster_u  = pp.cluster_dominante(hist_ids, cluster_map)
    hybrid     = pp.recomendar(base_recs, hist_ids, cluster_map, pool_saludable, cluster_u, k=K,
                              pool_con_sello=pool_con_sello)
    nd         = pp.evaluar(hybrid, hist_ids, cluster_map, pool_saludable, total_clusters, cluster_u)
    print(f'\nUsuario {uid} | Cluster: {cluster_u} | '
          f'Diversidad: {nd["Diversidad"]:.2f} | Novedad: {nd["Novedad"]:.2f}')
    for j, rid in enumerate(hybrid, 1):
        nm   = recipe_name_map.get(rid, rid)
        cat  = cluster_map.get(rid, '?')
        sel  = recipe_sellos.get(rid, 0)
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

# Recolectar recetas del agente CON sellos (α=0.5)
random.seed(42)
_ql_uids = random.sample(list(user_interactions_test.keys()), min(500, len(user_interactions_test)))
_ql_rids = []
for _uid in _ql_uids:
    _th = user_interactions_train.get(_uid, [])
    _st = get_user_state(_th, recipe_sellos)
    _sa = {recipe_to_action[r] for r, _ in _th if r in recipe_to_action}
    _ql_rids.extend([action_to_recipe[a]
                     for a in agent_con.get_top_k(_st, k=K, exclude_actions=_sa)])

# Venn: intersección de sellos en recetas recomendadas
_ql_rf = recipes_f[recipes_f['RecipeId'].isin(set(_ql_rids))]
_hcols  = ['IsHighSaturatedFat', 'IsHighSugar', 'IsHighSodium', 'IsHighCalories']
_hnames = ['Grasas Saturadas', 'Azúcares', 'Sodio', 'Calorías']
_venn_sets_ql = {n: set(_ql_rf[_ql_rf[c] == True]['RecipeId']) for c, n in zip(_hcols, _hnames)}
plt.figure(figsize=(10, 10))
_venn_fn(_venn_sets_ql)
plt.title('Q-Learning — Intersección de Sellos en Recetas Recomendadas (α=0.5)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_venn_sellos.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: ql_venn_sellos.png')

# Pareto: puntos de la sensitivity_df (ndcg vs ss_at_k para cada α)
_ql_pts = list(zip(
    [f'α={row.alpha}' for _, row in sensitivity_df.iterrows()],
    sensitivity_df['ndcg'].tolist(),
    sensitivity_df['ss_at_k'].tolist(),
))
_ql_cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(_ql_pts)))
fig, ax = plt.subplots(figsize=(8, 6))
for (_lbl, _nd, _ss), _col in zip(_ql_pts, _ql_cols):
    ax.scatter([_nd], [_ss], s=160, color=_col, zorder=5, edgecolors='black', lw=1.2)
    ax.annotate(_lbl, xy=(_nd, _ss), xytext=(5, 4), textcoords='offset points', fontsize=9)
_spts = sorted(_ql_pts, key=lambda x: x[1])
_px, _py, _mx = [], [], -1.0
for _, _nd, _ss in _spts:
    if _ss > _mx:
        _px.append(_nd); _py.append(_ss); _mx = _ss
if len(_px) > 1:
    ax.plot(_px, _py, '--', color='navy', lw=1.8, alpha=0.7, label='Frontera Pareto')
    ax.legend(fontsize=9)
ax.set_xlabel('nDCG@10 ↑  (Relevancia)', fontsize=12)
ax.set_ylabel('SS@10 ↑  (% sin sellos)', fontsize=12)
ax.set_title('Q-Learning — Trade-off Relevancia vs Salud por α', fontsize=12)
ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_pareto.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: ql_pareto.png')

# Clusters: distribución de categorías en recetas recomendadas
_ql_cats = pd.Series([cluster_map.get(rid, 'Unknown') for rid in _ql_rids]).value_counts().head(20)
fig, ax = plt.subplots(figsize=(14, 6))
_ql_cats.plot(kind='bar', ax=ax, color='darkorange', edgecolor='white')
ax.set_title('Q-Learning — Top-20 Clusters en Recetas Recomendadas (α=0.5)', fontsize=13)
ax.set_xlabel('Categoría (Cluster)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ql_clusters.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Guardado: ql_clusters.png')

print('\n¡Listo!')
