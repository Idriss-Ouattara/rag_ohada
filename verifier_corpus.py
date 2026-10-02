"""
Vérification qualité de corpus_ohada.json — à lancer après scrape_ohada.py.

Ce script ne modifie rien : il inspecte le corpus et affiche des signaux
d'alerte. L'idée est de repérer les problèmes AVANT l'indexation, parce
qu'un mauvais découpage propagé dans l'indexation coûte plus cher à
corriger après coup (il faut ré-indexer).

Ce qu'on vérifie, et pourquoi :

1. Nombre total d'articles + répartition par acte
   -> Un chiffre très bas par rapport à ce qu'on attend (ex. 10 articles
      pour un texte qui en a normalement 200+) signale un problème de regex.

2. Articles avec un texte suspicieusement court (< 20 caractères)
   -> Souvent le signe d'un découpage cassé : deux "Article N" collés sans
      contenu entre eux.

3. Bruit résiduel (URL, "page X / Y") encore présent dans le texte
   -> Le nettoyage des en-têtes/pieds de page n'a pas tout absorbé pour ce
      document (mise en forme différente d'un acte à l'autre).

4. Doublons de numéro d'article au sein d'un même acte
   -> Ne devrait normalement jamais arriver ; si oui, la regex a capturé
      un faux positif quelque part (ex. une référence "voir article 12"
      confondue avec un titre d'article).

Utilisation :
    python verifier_corpus.py
"""

import json
import re
from collections import Counter
from pathlib import Path

FICHIER_CORPUS = Path(__file__).parent / "corpus_ohada.json"

MOTIFS_BRUIT = [
    re.compile(r"https?://\S+"),
    re.compile(r"page\s+\d+\s*/\s*\d+", re.IGNORECASE),
]


def charger_corpus() -> list[dict]:
    if not FICHIER_CORPUS.exists():
        raise SystemExit(
            f"Fichier introuvable : {FICHIER_CORPUS}. "
            "Lance d'abord scrape_ohada.py dans ce même dossier."
        )
    return json.loads(FICHIER_CORPUS.read_text(encoding="utf-8"))


def verifier(articles: list[dict]) -> None:
    print(f"Total articles : {len(articles)}\n")

    # --- 1. Répartition par acte ---
    par_acte = Counter(a["acte_id"] for a in articles)
    print("Répartition par acte :")
    for acte_id, nb in par_acte.items():
        print(f"  {acte_id} : {nb} articles")
    print()

    # --- 2. Articles trop courts ---
    trop_courts = [a for a in articles if len(a["texte"]) < 20]
    print(f"Articles suspicieusement courts (< 20 caractères) : {len(trop_courts)}")
    for a in trop_courts[:5]:  # on n'affiche que les 5 premiers pour ne pas noyer
        print(f"  [{a['acte_id']}] Article {a['article_numero']} : {a['texte']!r}")
    print()

    # --- 3. Bruit résiduel ---
    avec_bruit = []
    for a in articles:
        if any(motif.search(a["texte"]) for motif in MOTIFS_BRUIT):
            avec_bruit.append(a)
    print(f"Articles avec bruit résiduel (URL, 'page X/Y') : {len(avec_bruit)}")
    for a in avec_bruit[:5]:
        print(f"  [{a['acte_id']}] Article {a['article_numero']}")
    print()

    # --- 4. Doublons de numéro au sein d'un même acte ---
    vus = Counter((a["acte_id"], a["article_numero"]) for a in articles)
    doublons = {cle: n for cle, n in vus.items() if n > 1}
    print(f"Doublons de numéro d'article : {len(doublons)}")
    for (acte_id, numero), n in list(doublons.items())[:5]:
        print(f"  [{acte_id}] Article {numero} apparaît {n} fois")
    print()

    # --- Aperçu de 2 articles au hasard, pour un contrôle visuel ---
    print("--- Aperçu (2 premiers articles) ---")
    for a in articles[:2]:
        print(f"\n[{a['acte_id']}] Article {a['article_numero']}")
        print(f"  Titre section    : {a['titre_section']}")
        print(f"  Chapitre section : {a['chapitre_section']}")
        print(f"  Texte : {a['texte'][:150]}...")


if __name__ == "__main__":
    articles = charger_corpus()
    verifier(articles)
    diagnostiquer_titres_figes(articles)

# --- Diagnostic ajoute : titre_section qui ne change jamais sur une longue plage ---
def diagnostiquer_titres_figes(articles):
    print("\n--- Diagnostic : titre_section fige sur une longue plage ---")
    dernier_titre = None
    compteur = 0
    for a in articles:
        if a['titre_section'] == dernier_titre:
            compteur += 1
        else:
            if compteur > 30:
                print(f"  '{dernier_titre}' repete sur {compteur} articles consecutifs (suspect)")
            dernier_titre = a['titre_section']
            compteur = 1
    if compteur > 30:
        print(f"  '{dernier_titre}' repete sur {compteur} articles consecutifs (suspect)")
