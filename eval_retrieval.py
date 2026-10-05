"""
Évaluation du retrieval — mesure si le système retrouve le bon article
pour des questions dont TU connais déjà la réponse.

Le principe : on construit une petite liste de cas de test, chacun avec
  - une question en langage naturel
  - le numéro de l'article qui devrait normalement répondre à cette question

On lance la recherche pour chaque question, et on vérifie si l'article
attendu apparaît dans les top_k résultats. Ça donne un score chiffré
(ex. 8/10) et, surtout, ça isole clairement deux causes d'échec bien
différentes :
  - le système ne retrouve pas un article qui EST dans le corpus (vrai
    problème de retrieval, à corriger : chunking, alpha, etc.)
  - le système ne retrouve rien de bon parce que l'article n'est
    simplement PAS dans le corpus (pas un bug, juste un trou à combler
    plus tard en scrapant d'autres actes uniformes)

Utilisation :
    python eval_retrieval.py
"""
import sys
from indexer import (
    charger_corpus,
    construire_index_bm25,
    construire_ou_charger_embeddings,
    rechercher,
)
from sentence_transformers import SentenceTransformer

# NOM_MODELE = "paraphrase-multilingual-MiniLM-L12-v2"
NOM_MODELE = "paraphrase-multilingual-MiniLM-L12-v2"
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35

CAS_DE_TEST = [
    {
        "question": "Quelles sociétés sont soumises à l'Acte uniforme sur les sociétés commerciales ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "1",
    },
    {
        "question": "Quelles clauses statutaires sont interdites si elles vont à l'encontre de l'Acte uniforme ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "2",
    },
    {
        "question": "Les actes accomplis avant la constitution de la société peuvent-ils être repris par elle ensuite ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "108",
    },
    {
        "question": "Comment la loi définit-elle la société commerciale ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "4",
    },
    {
        "question": "Une société commerciale peut-elle être créée par une seule personne ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "5",
    },
    {
        "question": "Les associés peuvent-ils conclure des conventions en dehors des statuts pour organiser leurs relations ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "2-1",
    },
    {
        "question": "Quel est le montant minimum du capital social d'une société à responsabilité limitée ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "311",
    },
    {
        "question": "Un gérant associé peut-il voter sur la décision fixant sa propre rémunération ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "325",
    },
    {
        "question": "Comment les décisions collectives des associés peuvent-elles être prises ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "133",
    },
    {
        "question": "Qui établit les états financiers de synthèse à la clôture de l'exercice ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "137",
    },
    {
        "question": "Quelles sont les causes qui mettent fin à une société commerciale ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "200",
    },
    {
        "question": "Quelle majorité est exigée pour la cession globale de l'actif dans une société en nom collectif ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "215",
    },
    {
        "question": "Pour quelles causes une société anonyme peut-elle être dissoute ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "736",
    },
    {
        "question": "Les administrateurs d'une société anonyme sont-ils responsables des fautes commises dans leur gestion ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "740",
    },
    {
        "question": "Quelle sanction encourent les dirigeants qui n'ont pas fait désigner de commissaire aux comptes ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "897",
    },
    {
        "question": "Combien de temps après sa publication au Journal Officiel l'Acte uniforme entre-t-il en vigueur ?",
        "acte_id_attendu": "audscgie",
        "article_attendu": "920",
    },
]

TOP_K = 5  # on vérifie si l'article attendu est dans les 5 premiers résultats


def evaluer():
    articles = charger_corpus()
    index_bm25 = construire_index_bm25(articles)

    print(f"[embeddings] chargement du modèle {NOM_MODELE}...")
    modele = SentenceTransformer(NOM_MODELE)
    embeddings, morceau_vers_article = construire_ou_charger_embeddings(articles, modele)

    nb_reussis = 0
    print(f"\n--- Évaluation sur {len(CAS_DE_TEST)} cas ---\n")

    for cas in CAS_DE_TEST:
        resultats = rechercher(
            cas["question"], articles, index_bm25, embeddings, morceau_vers_article,
            # modele, top_k=TOP_K, alpha=0.35,  # test : plus de poids au sémantique (défaut 0.5)
            modele, top_k=TOP_K, alpha=ALPHA,
        )

        # On regarde si l'article attendu apparaît quelque part dans le top_k,
        # et à quel rang exactement (1er, 2e, ...) pour voir si c'est net ou limite.
        rang_trouve = None
        for rang, r in enumerate(resultats, start=1):
            if r["acte_id"] == cas["acte_id_attendu"] and r["article_numero"] == cas["article_attendu"]:
                rang_trouve = rang
                break

        if rang_trouve:
            nb_reussis += 1
            print(f"[OK]   rang {rang_trouve} — \"{cas['question']}\"")
        else:
            print(f"[RATÉ] article {cas['article_attendu']} absent du top {TOP_K} — \"{cas['question']}\"")
            print(f"       articles retournés à la place : "
                  f"{[r['article_numero'] for r in resultats]}")

    print(f"\n--- Score : {nb_reussis}/{len(CAS_DE_TEST)} ---")


if __name__ == "__main__":
    evaluer()
