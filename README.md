# OHADA RAG

Assistant juridique **sourcé** sur le droit OHADA : on pose une question en français, le système retrouve les articles pertinents dans les Actes uniformes, puis un LLM local rédige une réponse qui **cite chaque article** utilisé, ou refuse de répondre si les articles ne suffisent pas.

## Pourquoi ce projet ?

Un LLM généraliste connaît mal le droit OHADA et peut inventer un article plausible. Ici, la réponse est contrainte par les textes réellement retrouvés, avec un prompt qui impose la citation `(Article N)` et le refus explicite en cas de manque d'information.

## Architecture

```
PDF des Actes uniformes
   │  scraping + nettoyage + découpage en articles
   ▼
corpus_ohada.json (1 entrée = 1 article, avec livre / titre / chapitre)
   │
   ├──► Index BM25 (tokenisation + stemming français)
   └──► Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
                │
Question ───────┴──► scores BM25 + scores cosinus ──► fusion pondérée (alpha) ──► top-k articles
                                                                                      │
                                                                  Mistral (Ollama, local) + prompt strict
                                                                                      ▼
                                                                         Réponse avec citations (Article N)
```

## Choix techniques

| Brique | Choix | Pourquoi |
|---|---|---|
| Extraction PDF | `pdfplumber` | Texte page par page, puis nettoyage des en-têtes et pieds de page |
| Découpage | Regex sur « Article N » (numéros composés type `2-1` inclus) | Un article de loi = une unité de recherche naturelle |
| Contexte hiérarchique | Livre + Titre + Chapitre ajoutés au texte indexé | Un article ne répète pas son contexte (l'article sur le capital minimum d'une SARL ne contient pas le mot « SARL ») |
| Recherche lexicale | BM25 (`rank_bm25`) + `FrenchStemmer` (NLTK) | Bon sur le vocabulaire juridique exact ; le stemming gère les accords et conjugaisons |
| Recherche sémantique | `paraphrase-multilingual-MiniLM-L12-v2`, similarité cosinus (NumPy) | Capte le sens sans mot commun |
| Articles longs | Découpage par phrases en morceaux ≤ limite de tokens du modèle, **meilleur score** retenu par article | Évite la troncature silencieuse des articles longs |
| Cache | Embeddings en `.npy` + empreinte SHA-256 du corpus | Réindexation automatique si le corpus change |
| Fusion | Scores min-max normalisés, `alpha·BM25 + (1-alpha)·sémantique` | Simple, réglable, interprétable |
| Génération | Mistral via Ollama (local, gratuit) | Aucune clé API, aucune donnée envoyée à l'extérieur |

## Corpus

- Acte uniforme relatif au droit des sociétés commerciales et du GIE (AUSCGIE)
- Acte uniforme portant sur le droit commercial général (AUDCG)
- **1 396 articles** au total, chacun avec `acte_id`, `article_numero`, `livre_section`, `titre_section`, `chapitre_section`, `texte`
- `verifier_corpus.py` contrôle la qualité : répartition par acte, textes trop courts, bruit résiduel, doublons

## Installation

```bash
git clone https://github.com/Idriss-Ouattara/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
```

Pour la génération : installer [Ollama](https://ollama.com) puis `ollama pull mistral`.

## Utilisation

```bash
python scrape_ohada.py       # télécharge les PDF et construit corpus_ohada.json
python verifier_corpus.py     # contrôle qualité du corpus
python indexer.py             # recherche interactive (calcule les embeddings au 1er lancement)
python generate_local.py     # questions-réponses avec citations
python eval_retrieval.py      # évaluation de la récupération
```

## Comprendre les scores

Quand on lance `indexer.py`, chaque résultat affiche trois valeurs :

```
[Article 311] score=0.83 (lexical=6.0, sémantique=0.62)
```

| Score | Ce qu'il mesure | Échelle |
|---|---|---|
| `lexical` (BM25) | Les mots de la question apparaissent-ils dans l'article ? Un mot rare pèse plus qu'un mot courant, et un article très long est légèrement pénalisé. | Sans plafond (3, 12, 25…) : seule la comparaison entre articles a un sens |
| `sémantique` (cosinus) | L'article parle-t-il de la même chose que la question, même avec d'autres mots ? Pour un article long, on garde le meilleur de ses morceaux. | Proche de 1 : même sens ; proche de 0 : sans rapport |
| `score` (combiné) | Mélange des deux, après les avoir ramenés chacun sur une échelle de 0 à 1 (le pire article vaut 0, le meilleur vaut 1) | 0 à 1 |

Formule : `score = alpha × BM25 normalisé + (1 − alpha) × sémantique normalisé`. Avec `alpha = 1`, on ne garde que BM25 ; avec `alpha = 0`, que le sémantique.

### Exemple chiffré

Trois articles pour une même question, avec `alpha = 0,35` :

| | BM25 brut | Cosinus brut | BM25 normalisé | Cosinus normalisé | Score combiné |
|---|---|---|---|---|---|
| A | 12,0 | 0,40 | 1,00 | 0,31 | 0,35×1 + 0,65×0,31 ≈ **0,55** |
| B | 6,0 | 0,62 | 0,50 | 1,00 | 0,35×0,5 + 0,65×1 = **0,83** |
| C | 0,0 | 0,30 | 0,00 | 0,00 | **0** |

B passe devant : il est moyen sur les mots mais très bon sur le sens, et `alpha = 0,35` donne plus de poids au sens. Avec `alpha = 1`, A l'emporterait.

### Attention : le score combiné est relatif

Un score de 0,83 ne veut pas dire « 83 % de confiance », mais « le meilleur parmi ces résultats ». Le système renvoie toujours ses `top_k` meilleurs articles, même si aucun ne répond à la question. C'est pourquoi le refus de répondre est géré dans le prompt de génération (`generate_local.py`) et non dans la recherche.

## Évaluation

`eval_retrieval.py` mesure si la récupération retrouve le bon article : **16 questions** rédigées à la main sur l'AUSCGIE, chacune associée à l'article attendu. Métrique : **Hit@5** (l'article attendu figure-t-il dans les 5 premiers résultats ?), avec affichage du rang.

| Configuration | Hit@5 (/16) |
|---|---|
| BM25 seul (`alpha=1`) | `…` |
| Sémantique seul (`alpha=0`) | `…` |
| Hybride (`alpha=0.35`) | `…` |

Le script distingue aussi deux types d'échecs : un article présent dans le corpus mais non retrouvé (problème de récupération) et un article absent du corpus (trou de couverture).

## Limites et pistes

- Évaluation limitée à l'AUSCGIE (16 questions) ; l'AUDCG n'est pas encore couvert par les cas de test
- Pas encore de reranker ni d'évaluation automatique de la fidélité des réponses générées
- Autres Actes uniformes à ajouter (sûretés, recouvrement, procédures collectives, arbitrage, comptabilité)
- Pistes : modèle d'embeddings plus puissant, fusion par rangs (RRF), reranking

## Auteur

**Idriss Ouattara**, Master 2 NLP, Université Grenoble Alpes ·
[GitHub](https://github.com/Idriss-Ouattara)
