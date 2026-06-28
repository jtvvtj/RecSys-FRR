"""
RecommenderPostProcessor
========================
Estrategia híbrida de recomendación + métricas de Diversidad y Novedad
orientadas a salud. Diseñado para ser importado desde cualquier script de réplica.

Diversidad : clusters únicos en las recomendaciones / k
Novedad    : % de recs que (1) no fueron vistas, (2) son saludables,
             (3) pertenecen al cluster dominante del usuario
"""
import re
import random
from collections import Counter

_WEIGHT_VOL_RE = re.compile(
    r'\b(gram|grams|kg|kilogram|oz|ounce|pound|lbs?|cup|cups|'
    r'tablespoon|tbsp|teaspoon|tsp|liters?|litres?|ml|quarts?|pints?)\b',
    re.IGNORECASE,
)


def fix_serving_sizes(df):
    """
    Corrige ExtractedServingSize para recetas donde ConsolidatedServings=1
    pero RecipeYield indica múltiples unidades (ej. '24 bites', '16 snacks').
    En esos casos ExtractedServingSize contiene el peso TOTAL del batch, no
    el peso por porción individual. La corrección divide por el número parseado.

    Requiere columnas: ExtractedServingSize (float, ya extraído), ConsolidatedServings,
    RecipeYield.
    """
    import pandas as pd

    df = df.copy()

    def _parse_n(yield_str):
        if pd.isna(yield_str):
            return 1
        s = str(yield_str).strip()
        if _WEIGHT_VOL_RE.search(s):
            return 1
        m = re.search(r'(\d+(?:\.\d+)?)\s*dozen', s, re.IGNORECASE)
        if m:
            return max(1, round(float(m.group(1)) * 12))
        m = re.search(r'^\s*(\d+)', s)
        if m:
            n = int(m.group(1))
            return n if n > 1 else 1
        return 1

    mask = df['ConsolidatedServings'] == 1
    n_series = df.loc[mask, 'RecipeYield'].apply(_parse_n)
    df.loc[mask, '_fix_n'] = n_series

    apply_mask = (
        mask
        & (df['_fix_n'] > 1)
        & (df['ExtractedServingSize'] / df['_fix_n'] >= 5)
    )
    df.loc[apply_mask, 'ExtractedServingSize'] = (
        df.loc[apply_mask, 'ExtractedServingSize'] / df.loc[apply_mask, '_fix_n']
    )
    df.drop(columns=['_fix_n'], errors='ignore', inplace=True)
    return df


class RecommenderPostProcessor:

    def recomendar(self, recomendaciones_base, historial_usuario,
                   cluster_map, pool_saludable, cluster_usuario, k=10,
                   n_con_sello=7, pool_con_sello=None):
        """
        Garantiza n_con_sello de k recomendaciones con >=1 sello (relevancia real)
        y (k - n_con_sello) sin sellos como ruido saludable.

        pool_con_sello: precomputar fuera del loop para mayor eficiencia.
                        Si es None se calcula aquí (más lento en loops grandes).

        Prioridad para los slots con sello:
          1. Recs del modelo base que ya tienen sello
          2. Recs del pool con sello del mismo cluster del usuario
          3. Cualquier rec con sello del pool general

        Prioridad para los slots sin sello (ruido):
          1. Recs del modelo base sin sello
          2. pool_saludable del mismo cluster
          3. Cualquier rec del pool_saludable
        """
        pool_sano = set(pool_saludable)
        pool_con  = pool_con_sello if pool_con_sello is not None else set(cluster_map.keys()) - pool_sano
        hist_set  = set(historial_usuario)
        seen      = set(hist_set)
        n_ruido   = k - n_con_sello

        # ── 1. Separar recs del modelo ────────────────────────────────────────
        model_con = [r for r in recomendaciones_base if r not in seen and r in pool_con]
        model_sin = [r for r in recomendaciones_base if r not in seen and r in pool_sano]

        recs_con = model_con[:n_con_sello]
        seen.update(recs_con)
        recs_sin = [r for r in model_sin if r not in seen][:n_ruido]
        seen.update(recs_sin)

        # ── 2. Rellenar slots con sello desde el pool general ─────────────────
        if len(recs_con) < n_con_sello:
            needed = n_con_sello - len(recs_con)
            cands = [r for r in pool_con
                     if r not in seen and cluster_map.get(r) == cluster_usuario]
            random.shuffle(cands)
            fill = cands[:needed]
            if len(fill) < needed:
                extra = [r for r in pool_con if r not in seen and r not in set(fill)]
                random.shuffle(extra)
                fill += extra[:needed - len(fill)]
            recs_con += fill
            seen.update(fill)

        # ── 3. Rellenar slots de ruido (sin sello) desde pool_saludable ───────
        if len(recs_sin) < n_ruido:
            needed = n_ruido - len(recs_sin)
            cands = [r for r in pool_saludable
                     if r not in seen and cluster_map.get(r) == cluster_usuario]
            random.shuffle(cands)
            fill = cands[:needed]
            if len(fill) < needed:
                extra = [r for r in pool_saludable if r not in seen and r not in set(fill)]
                random.shuffle(extra)
                fill += extra[:needed - len(fill)]
            recs_sin += fill

        result = recs_con + recs_sin
        random.shuffle(result)
        return result[:k]

    def diversidad(self, recomendaciones, cluster_map, total_clusters=None):
        """
        Clusters únicos en la lista / k.
        Rango [0, 1]: 1 = máxima diversidad (cada item de un cluster distinto).
        """
        k = len(recomendaciones)
        if k == 0:
            return 0.0
        clusters = [cluster_map.get(rid) for rid in recomendaciones
                    if cluster_map.get(rid) is not None]
        return len(set(clusters)) / k if clusters else 0.0

    def novedad(self, recomendaciones, historial_usuario,
                pool_saludable, cluster_map, cluster_usuario):
        """
        Proporción de recetas que cumplen las 3 condiciones:
          1) No vista por el usuario
          2) Saludable (en pool_saludable)
          3) Del cluster dominante del usuario
        """
        if not recomendaciones:
            return 0.0
        pool_set = set(pool_saludable)
        hist_set = set(historial_usuario)
        novel = sum(
            1 for rid in recomendaciones
            if rid not in hist_set
            and rid in pool_set
            and cluster_map.get(rid) == cluster_usuario
        )
        return novel / len(recomendaciones)

    def evaluar(self, recomendaciones, historial_usuario, cluster_map,
                pool_saludable, total_clusters, cluster_usuario):
        return {
            'Diversidad': self.diversidad(recomendaciones, cluster_map, total_clusters),
            'Novedad':    self.novedad(recomendaciones, historial_usuario,
                                       pool_saludable, cluster_map, cluster_usuario),
        }

    @staticmethod
    def cluster_dominante(historial_usuario, cluster_map):
        """Retorna el cluster más frecuente en el historial del usuario."""
        clusters = [cluster_map.get(rid) for rid in historial_usuario
                    if cluster_map.get(rid) is not None]
        if not clusters:
            return None
        return Counter(clusters).most_common(1)[0][0]
