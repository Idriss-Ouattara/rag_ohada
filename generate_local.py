"""
Génération de réponses avec citation — version Ollama (LLM local, gratuit).

Même principe que generate.py, mais on parle à Ollama (qui tourne en local
sur ta machine, http://localhost:11434) au lieu de l'API Anthropic. Pas de
clé API, pas de coût, mais il faut qu'Ollama tourne et que le modèle soit
déjà téléchargé (voir instructions données à côté de ce fichier).

Pourquoi les mêmes règles strictes que generate.py : le risque
d'hallucination (le LLM qui invente une réponse plausible) existe pour
n'importe quel modèle, local ou via API — le prompt verrouille ce
comportement de la même façon dans les deux cas.

Utilisation :
    1. Installer Ollama (ollama.com) et lancer : ollama pull mistral
    2. pip install requests
    3. python generate_local.py
"""

import requests

from indexer import (
    charger_corpus,
    construire_index_bm25,
    construire_ou_charger_embeddings,
    rechercher,
)
from sentence_transformers import SentenceTransformer

NOM_MODELE_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
MODELE_LLM = "mistral"  # doit correspondre à un modèle déjà téléchargé (ollama pull ...)
URL_OLLAMA = "http://localhost:11434/api/chat"
NB_ARTICLES_CONTEXTE = 6

PROMPT_SYSTEME = """Tu es un assistant juridique spécialisé en droit OHADA.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des articles fournis ci-dessous. N'utilise
   aucune connaissance juridique extérieure, même si tu penses la connaître.
2. Pour CHAQUE affirmation juridique, cite le numéro de l'article exact
   sur lequel elle s'appuie, au format (Article N).
3. Si les articles fournis ne permettent pas de répondre à la question,
   dis-le explicitement : "Les articles fournis ne permettent pas de
   répondre à cette question." Ne comble jamais un manque d'information
   par une supposition.
4. Reste concis et précis, dans le registre d'un professionnel du droit."""


def construire_contexte(articles_trouves: list[dict]) -> str:
    blocs = []
    for a in articles_trouves:
        blocs.append(
            f"[Article {a['article_numero']} — {a['acte_titre']}]\n{a['texte']}"
        )
    return "\n\n".join(blocs)


def generer_reponse(question: str, articles_trouves: list[dict]) -> str:
    contexte = construire_contexte(articles_trouves)

    prompt_utilisateur = f"""Articles disponibles :

{contexte}

---

Question : {question}"""

    reponse = requests.post(
        URL_OLLAMA,
        json={
            "model": MODELE_LLM,
            "messages": [
                {"role": "system", "content": PROMPT_SYSTEME},
                {"role": "user", "content": prompt_utilisateur},
            ],
            "stream": False,  # on veut la réponse complète d'un coup, pas en flux
        },
        timeout=300,  # un modèle local peut être lent, surtout au premier
                      # chargement en mémoire (jusqu'à 1-2 min avant même de
                      # commencer à générer)
    )
    reponse.raise_for_status()
    return reponse.json()["message"]["content"]


def main():
    # Vérification rapide qu'Ollama tourne bien, avec un message clair sinon
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "Impossible de joindre Ollama sur http://localhost:11434. "
            "Vérifie qu'Ollama est bien lancé (l'installation le démarre "
            "normalement automatiquement en arrière-plan)."
        )

    articles = charger_corpus()
    index_bm25 = construire_index_bm25(articles)

    print(f"[embeddings] chargement du modèle {NOM_MODELE_EMBEDDINGS}...")
    modele_embeddings = SentenceTransformer(NOM_MODELE_EMBEDDINGS)
    embeddings, morceau_vers_article = construire_ou_charger_embeddings(articles, modele_embeddings)

    print("\nTape une question juridique (ou 'quit' pour sortir) :")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        articles_trouves = rechercher(
            question, articles, index_bm25, embeddings, morceau_vers_article,
            modele_embeddings, top_k=NB_ARTICLES_CONTEXTE,
        )

        print("\n[recherche] articles utilisés comme contexte :",
              [a["article_numero"] for a in articles_trouves])

        print("\n[génération en cours, peut prendre 10-30s en local...]\n")
        reponse = generer_reponse(question, articles_trouves)
        print(reponse)


if __name__ == "__main__":
    main()
