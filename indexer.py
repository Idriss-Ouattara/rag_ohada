"""
Indexation hybride du corpus OHADA — BM25 (lexical) + embeddings (sémantique).

Ce que fait ce script :
  1. Charge corpus_ohada.json.
  2. Construit un index BM25 (recherche par mots-clés).
  3. Calcule un embedding (vecteur de sens) pour chaque article.
  4. Pour une question donnée, combine les deux scores et renvoie les
     articles les plus pertinents.

Pourquoi combiner les deux (rappel du principe) :
  - BM25 = recherche lexicale. Fort sur le vocabulaire juridique exact
    ("dissolution", "gérant", "SARL"), aveugle aux synonymes.
  - Embeddings = recherche sémantique. Capture le sens même sans mot
    commun, mais peut remonter un article "dans le bon thème" sans viser
    le bon article précis.
  - En combinant, on couvre les deux types d'erreurs.

Utilisation :
    pip install sentence-transformers rank_bm25 numpy nltk
    python indexer.py

La première exécution télécharge le modèle d'embeddings (~470 Mo, une
seule fois, mis en cache ensuite).
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from nltk.stem.snowball import FrenchStemmer

FICHIER_CORPUS = Path(__file__).parent / "corpus_ohada.json"
FICHIER_EMBEDDINGS = Path(__file__).parent / "embeddings_ohada.npy"
# Fichier "compagnon" des embeddings : garde la trace de quel morceau de
# texte appartient à quel article (nécessaire depuis qu'un article peut
# être découpé en plusieurs morceaux, voir section 3 plus bas) et une
# empreinte du corpus utilisé, pour savoir si le cache est encore valide.
FICHIER_EMBEDDINGS_META = Path(__file__).parent / "embeddings_ohada_meta.json"

# Modèle multilingue : gère le français juridique correctement, léger
# donc raisonnable à faire tourner sur un ordinateur portable.
NOM_MODELE = "paraphrase-multilingual-MiniLM-L12-v2"

# 1. CHARGEMENT DU CORPUS

def charger_corpus() -> list[dict]:
    if not FICHIER_CORPUS.exists():
        raise SystemExit(
            f"Fichier introuvable : {FICHIER_CORPUS}. "
            "Lance d'abord scrape_ohada.py dans ce même dossier."
        )
    return json.loads(FICHIER_CORPUS.read_text(encoding="utf-8"))


"""2. INDEX LEXICAL (BM25)
BM25 a besoin du texte découpé en mots ("tokenisé"), pas du texte brut.
On fait une tokenisation simple : minuscules + découpage sur tout ce qui
n'est pas une lettre. Pas besoin d'une librairie NLP complète pour ça."""


def tokeniser(texte: str) -> list[str]:
    """Découpe en mots ET les ramène à leur radical commun (stemming).

    Pourquoi le stemming est nécessaire ici : le français conjugue et
    accorde énormément ("déterminé" / "détermine" / "déterminent" pour un
    même verbe). Sans stemming, BM25 traite ces formes comme des mots
    totalement différents et rate des correspondances évidentes pour un
    humain. Le stemmer ramène toutes ces formes à un radical commun
    ("détermin"), donc "détermine" dans une question matche bien
    "déterminé" dans un article.
    """
    mots = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", texte.lower())
    return [_stemmer.stem(mot) for mot in mots]


_stemmer = FrenchStemmer()


def texte_indexable(article: dict) -> str:
    """Texte réellement utilisé pour la recherche (BM25 + embeddings).

    Un article ne répète pas forcément le contexte qui le rend pertinent :
    ex. l'article sur le capital minimum d'une SARL ne contient jamais le
    mot "SARL" dans son propre texte — cette info n'existe que dans le
    titre de section ("Titre 1 - Constitution de la société à
    responsabilité limitée"). Sans ce préfixe, une question mentionnant
    "SARL" ne peut matcher cet article que par coïncidence, et se fait
    concurrencer par d'autres articles parlant aussi de "capital minimum"
    pour d'autres formes de société.
    """
    contexte = " — ".join(
        p for p in (article.get("titre_section"), article.get("chapitre_section")) if p
    )
    return f"{contexte}\n{article['texte']}" if contexte else article["texte"]


def construire_index_bm25(articles: list[dict]) -> BM25Okapi:
    corpus_tokenise = [tokeniser(texte_pour_recherche(a)) for a in articles]
    return BM25Okapi(corpus_tokenise)



# 2 bis. EMPREINTE DU CORPUS
#
# Sert à savoir si des embeddings déjà calculés sur le disque correspondent
# encore au corpus actuel. Se fier seulement à "le fichier existe" est
# dangereux : si corpus_ohada.json change (nouvel Acte ajouté, article
# recorrigé, etc.) sans supprimer embeddings_ohada.npy à la main, on se
# retrouverait à utiliser des vecteurs qui ne correspondent plus aux bons
# articles - une désynchronisation silencieuse, donc particulièrement
# dangereuse.


def calculer_hash_corpus(articles: list[dict]) -> str:
    contenu = "||".join(
        f"{a['acte_id']}#{a['article_numero']}#{a.get('livre_section') or ''}#"
        f"{a.get('titre_section') or ''}#{a.get('chapitre_section') or ''}#{a['texte']}"
        for a in articles
    )
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()



def texte_pour_recherche(article: dict) -> str:
    morceaux_contexte = [
        article.get("livre_section"), article.get("titre_section"), article.get("chapitre_section"),
    ]
    prefixe = " - ".join(m for m in morceaux_contexte if m)
    return f"{prefixe}. {article['texte']}" if prefixe else article["texte"]


def decouper_pour_embedding(texte: str, modele: SentenceTransformer, marge_tokens: int = 10) -> list[str]:
    """Découpe un texte en morceaux qui tiennent dans la limite de tokens du modèle.

    On découpe phrase par phrase (jamais au milieu d'une phrase, pour ne
    pas casser le sens) en regroupant les phrases tant que ça tient sous
    la limite. marge_tokens laisse de la place aux tokens spéciaux que le
    tokenizer ajoute lui-même ([CLS]/[SEP]).
    """
    limite = modele.max_seq_length - marge_tokens
    phrases = re.split(r"(?<=[.;:])\s+", texte.strip())

    morceaux = []
    morceau_courant = ""
    for phrase in phrases:
        candidat = f"{morceau_courant} {phrase}".strip() if morceau_courant else phrase
        nb_tokens = len(modele.tokenizer.encode(candidat, add_special_tokens=False))
        if nb_tokens <= limite or not morceau_courant:
            morceau_courant = candidat
        else:
            morceaux.append(morceau_courant)
            morceau_courant = phrase

    if morceau_courant:
        morceaux.append(morceau_courant)
    return morceaux or [texte]  # filet de sécurité (texte vide, cas limite)


def construire_ou_charger_embeddings(
    articles: list[dict], modele: SentenceTransformer
) -> tuple[np.ndarray, list[int]]:
    """Renvoie (embeddings, morceau_vers_article).

    morceau_vers_article[i] donne l'index (dans `articles`) de l'article
    auquel appartient le morceau i des embeddings — nécessaire pour
    ramener les scores calculés par morceau au niveau de l'article dans
    la fonction `rechercher`.
    """
    hash_actuel = calculer_hash_corpus(articles)

    if FICHIER_EMBEDDINGS.exists() and FICHIER_EMBEDDINGS_META.exists():
        meta = json.loads(FICHIER_EMBEDDINGS_META.read_text(encoding="utf-8"))
        if meta.get("corpus_hash") == hash_actuel:
            print("[cache] embeddings déjà calculés et à jour, chargement depuis le disque")
            return np.load(FICHIER_EMBEDDINGS), meta["morceau_vers_article"]
        print("[cache] corpus modifié depuis le dernier calcul -> réindexation complète")

    morceaux_texte: list[str] = []
    morceau_vers_article: list[int] = []
    for i, a in enumerate(articles):
        for morceau in decouper_pour_embedding(texte_pour_recherche(a), modele):
            morceaux_texte.append(morceau)
            morceau_vers_article.append(i)

    nb_articles_decoupes = len(morceaux_texte) - len(articles)
    print(f"[embeddings] {len(morceaux_texte)} morceaux à calculer pour {len(articles)} articles "
          f"({nb_articles_decoupes} morceaux supplémentaires dus au découpage des articles longs)")
    vecteurs = modele.encode(morceaux_texte, show_progress_bar=True, batch_size=32)

    np.save(FICHIER_EMBEDDINGS, vecteurs)
    FICHIER_EMBEDDINGS_META.write_text(
        json.dumps({"corpus_hash": hash_actuel, "morceau_vers_article": morceau_vers_article}),
        encoding="utf-8",
    )
    print(f"[ok] embeddings sauvegardés -> {FICHIER_EMBEDDINGS.name}")
    return vecteurs, morceau_vers_article


def similarite_cosinus(vecteur: np.ndarray, matrice: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre un vecteur et chaque ligne d'une matrice.

    Calculée à la main (pas de librairie dédiée) : c'est juste un produit
    scalaire normalisé, pas besoin de plus pour ce volume de données.
    """
    vecteur_norme = vecteur / np.linalg.norm(vecteur)
    matrice_normee = matrice / np.linalg.norm(matrice, axis=1, keepdims=True)
    return matrice_normee @ vecteur_norme



# 4. RECHERCHE HYBRIDE
#
# On normalise chaque score sur une échelle 0-1 (les scores BM25 et cosinus
# ne sont pas sur la même échelle au départ) avant de les additionner.
# alpha contrôle le poids relatif : alpha=0.35 donne un poids égal aux deux.


def normaliser(scores: np.ndarray) -> np.ndarray:
    minimum, maximum = scores.min(), scores.max()
    if maximum == minimum:  # évite une division par zéro si tous les scores sont égaux
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


def rechercher(
    question: str,
    articles: list[dict],
    index_bm25: BM25Okapi,
    embeddings: np.ndarray,
    morceau_vers_article: list[int],
    modele: SentenceTransformer,
    top_k: int = 5,
    alpha: float = 0.35,
) -> list[dict]:
    scores_bm25 = np.array(index_bm25.get_scores(tokeniser(question)))

    vecteur_question = modele.encode(question)
    scores_semantiques_par_morceau = similarite_cosinus(vecteur_question, embeddings)

    # Un article peut avoir plusieurs morceaux (découpage des articles
    # longs, voir decouper_pour_embedding) : on retient le MEILLEUR score
    # parmi ses morceaux, pas la moyenne. Si un seul passage d'un article
    # long est pertinent, l'article est pertinent — la moyenne noierait
    # ce signal dans le reste du texte, non pertinent pour la question.
    scores_semantiques = np.full(len(articles), -1.0)
    for i_morceau, i_article in enumerate(morceau_vers_article):
        score = scores_semantiques_par_morceau[i_morceau]
        if score > scores_semantiques[i_article]:
            scores_semantiques[i_article] = score

    scores_combines = (
        alpha * normaliser(scores_bm25)
        + (1 - alpha) * normaliser(scores_semantiques)
    )

    indices_tries = np.argsort(scores_combines)[::-1][:top_k]

    resultats = []
    for i in indices_tries:
        resultats.append({
            **articles[i],
            "score": float(scores_combines[i]),
            "score_lexical": float(scores_bm25[i]),
            "score_semantique": float(scores_semantiques[i]),
        })
    return resultats


# 5. POINT D'ENTRÉE — mode interactif simple pour tester


def main():
    articles = charger_corpus()
    print(f"{len(articles)} articles chargés\n")

    print("[bm25] construction de l'index lexical...")
    index_bm25 = construire_index_bm25(articles)

    print(f"[embeddings] chargement du modèle {NOM_MODELE}...")
    modele = SentenceTransformer(NOM_MODELE)
    embeddings, morceau_vers_article = construire_ou_charger_embeddings(articles, modele)

    print("\nTape une question juridique (ou 'quit' pour sortir) :")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        resultats = rechercher(question, articles, index_bm25, embeddings, morceau_vers_article, modele)
        for r in resultats:
            print(f"\n[Article {r['article_numero']}] score={r['score']:.2f} "
                  f"(lexical={r['score_lexical']:.1f}, sémantique={r['score_semantique']:.2f})")
            print(f"  {r['acte_titre']}")
            print(f"  {r['texte'][:200]}...")


if __name__ == "__main__":
    main()
