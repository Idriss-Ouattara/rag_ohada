"""
Scraper OHADA — script unique, à lancer tel quel.

Ce que fait ce script, dans l'ordre :
  1. Télécharge les PDF des Actes uniformes listés dans SOURCES (ci-dessous).
  2. Extrait le texte de chaque PDF.
  3. Découpe ce texte en articles de loi (un article = une entrée).
  4. Sauvegarde le tout dans un fichier corpus_ohada.json, à côté de ce script.

Utilisation :
    pip install requests pdfplumber
    python scrape_ohada.py

Tout se passe dans CE dossier — pas de sous-dossiers à créer. Deux
dossiers seront créés automatiquement au premier lancement :
  - pdfs/       : les PDF téléchargés (pour ne pas les retélécharger à chaque fois)
  - corpus_ohada.json : le résultat final, un article par entrée
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import requests
import pdfplumber


"""
1. LES SOURCES : la liste des textes à récupérer.
a Complèter au fur et à mesure pour ajouter un nouvel Acte uniforme
"""


SOURCES = [
    {
        "id": "audscgie",
        "titre": "Acte uniforme relatif au droit des sociétés commerciales et du GIE",
        "url": "https://faolex.fao.org/docs/pdf/oha195728.pdf",
    },
    {
        "id": "audcg",
        "titre": "Acte uniforme portant sur le droit commercial général",
        "url": "https://faolex.fao.org/docs/pdf/oha195620.pdf",
    },
]

DOSSIER_PDF = Path(__file__).parent / "pdfs"
FICHIER_SORTIE = Path(__file__).parent / "corpus_ohada.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NexaLing-OHADA-RAG/0.1; recherche academique)"
}



# 2. TÉLÉCHARGEMENT

def telecharger_pdf(source: dict) -> Path:
    """Télécharge un PDF s'il n'est pas déjà présent sur le disque."""
    chemin = DOSSIER_PDF / f"{source['id']}.pdf"

    if chemin.exists():
        print(f"[cache] {source['id']} déjà téléchargé, on saute")
        return chemin

    print(f"[dl] {source['id']} <- {source['url']}")
    reponse = requests.get(source["url"], headers=HEADERS, timeout=30)
    reponse.raise_for_status()  # erreur claire si le lien est mort (404, etc.)

    chemin.write_bytes(reponse.content)
    print(f"[ok] {chemin.name} ({len(reponse.content) / 1024:.0f} Ko)")
    return chemin


def extraire_texte_pdf(chemin_pdf: Path) -> str:
    """Lit toutes les pages d'un PDF et renvoie leur texte concaténé."""
    morceaux = []
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            morceaux.append(page.extract_text() or "")
    return "\n".join(morceaux)


""" 3. DÉCOUPAGE EN ARTICLES
Deux pièges réels rencontrés en testant sur le vrai PDF OHADA :
Chaque page du PDF répète un en-tête (titre de l'acte, date) et un pied
de page (URL, numéro de page). Extraites page par page, ces lignes se
retrouvent EN PLEIN MILIEU d'un article quand celui-ci est à cheval sur
deux pages. On les repère et on les retire : une ligne qui revient plus
de 3 fois dans le document n'est jamais du contenu juridique, un article
te pas mot pour mot.
Certains articles insérés lors de révisions ont un numéro composé,
ex. "Article 2-1". La regex doit accepter ce format.
"""


ARTICLE_PATTERN = re.compile(
    r"^(?:Article\s+(?P<num_a>\d+(?:-\d+)?)\s*$"
    r"|Art\.\s*(?P<num_b>\d+(?:-\d+)?)\s*\.\s*[-–]\s*)",
    re.MULTILINE,
)
# Deux formats rencontrés selon la source du PDF, d'où les deux alternatives
# ci-dessus :
#   - "Article 1"           (seul sur sa ligne)          -> AUSCGIE
#   - "Art.1.- Tout ..."    (numéro collé au texte)       -> AUDCG
# Le numéro capturé est dans num_a ou num_b selon le format matché (voir
# parse_articles : numero = m.group("num_a") or m.group("num_b")).
LIVRE_PATTERN = re.compile(r"^Livre\s+.+$", re.MULTILINE)
TITRE_PATTERN = re.compile(r"^Titre\s+.+$", re.MULTILINE)
CHAPITRE_PATTERN = re.compile(r"^Chapitre\s+.+$", re.MULTILINE)
SEUIL_REPETITION = 3

# Lignes qui ressemblent à un titre de section structurel : protégées de la
# suppression par fréquence même si elles se répètent verbatim ailleurs
# dans le document. Sans ça, un intitulé générique comme "Titre 1 -
# Dispositions générales", réutilisé tel quel au début de plusieurs Livres,
# finit par dépasser SEUIL_REPETITION et se fait supprimer comme s'il
# s'agissait d'un en-tête répété — alors que c'est un vrai titre de
# section, différent à chaque emplacement par le CONTEXTE (quel Livre le
# précède), même si le texte de la ligne elle-même est identique.
MOTIF_TITRE_STRUCTUREL = re.compile(
    r"^(Livre|Titre|Sous-titre|Chapitre|Section|Partie)\s+", re.IGNORECASE
)

# Ligne de pied de page type : "http://.../texte.html page 12 / 209"
# Elle contient un numéro de page qui change à CHAQUE page — donc elle
# n'est jamais identique mot pour mot d'une occurrence à l'autre, et le
# nettoyage par fréquence (ci-dessous) ne peut pas la voir comme répétée.
# On la retire donc séparément avec un motif, plutôt que par comptage.
#
# Le ".*" (plutôt que "\S+" pour l'URL) est nécessaire à cause d'un
# artefact d'extraction rencontré sur certains PDF : le texte justifié en
# bas de page ressort avec des espaces insérés à l'intérieur même de
# l'URL (ex. "economique.html" devient "e c o n o m i q u e . h tm l").
# Un "\S+" strict ratait donc silencieusement ces lignes.
MOTIF_PIED_DE_PAGE_URL = re.compile(r"^https?://.*page\s+\d+\s*/\s*\d+\s*$", re.IGNORECASE)

# Même problème pour l'en-tête répété du PDF de l'AUDCG (source
# Droit-Afrique.com) : la ligne "Acte uniforme OHADA portant sur ... N"
# se termine par le numéro de page, donc elle varie d'une occurrence à
# l'autre et échappe elle aussi au comptage de fréquence pur.
MOTIF_ENTETE_ACTE_PAGE = re.compile(r"^Acte uniforme\b.*\s\d{1,4}\s*$", re.MULTILINE)


def nettoyer_entetes_pieds_de_page(texte_brut: str) -> str:
    lignes = texte_brut.split("\n")

    # Étape A : retirer les lignes "url ... page X / Y" et les en-têtes à
    # numéro de page variable, qui échapperaient au comptage de fréquence
    # (voir les commentaires sur MOTIF_PIED_DE_PAGE_URL et
    # MOTIF_ENTETE_ACTE_PAGE plus haut).
    lignes = [
        l for l in lignes
        if not MOTIF_PIED_DE_PAGE_URL.match(l.strip())
        and not MOTIF_ENTETE_ACTE_PAGE.match(l.strip())
    ]

    # Étape B : retirer les lignes identiques répétées plus de N fois
    # (titre de l'acte, date d'adoption) — celles-ci SONT identiques d'une
    # page à l'autre, donc le comptage de fréquence les attrape bien. On
    # épargne les lignes qui ressemblent à un titre de section structurel
    # (voir MOTIF_TITRE_STRUCTUREL) : un intitulé générique réutilisé pour
    # plusieurs Livres n'est pas du bruit, même s'il se répète verbatim.
    compteur = Counter(
        ligne.strip() for ligne in lignes
        if ligne.strip() and not MOTIF_TITRE_STRUCTUREL.match(ligne.strip())
    )
    bruit = {ligne for ligne, n in compteur.items() if n > SEUIL_REPETITION}
    return "\n".join(l for l in lignes if l.strip() not in bruit)


def _derniere_correspondance_avant(pattern, texte, position):
    trouvailles = list(pattern.finditer(texte, 0, position))
    return trouvailles[-1].group(0).strip() if trouvailles else None


def parse_articles(texte_brut: str, acte_id: str, acte_titre: str) -> list[dict]:
    texte_propre = nettoyer_entetes_pieds_de_page(texte_brut)
    correspondances = list(ARTICLE_PATTERN.finditer(texte_propre))

    if not correspondances:
        print(f"[attention] aucun article détecté dans {acte_id} — à inspecter à la main")
        return []

    articles = []
    for i, m in enumerate(correspondances):
        numero = m.group("num_a") or m.group("num_b")
        debut = m.end()
        fin = correspondances[i + 1].start() if i + 1 < len(correspondances) else len(texte_propre)
        texte_article = re.sub(r"\n{2,}", "\n", texte_propre[debut:fin].strip())

        articles.append({
            "acte_id": acte_id,
            "acte_titre": acte_titre,
            "article_numero": numero,
            "livre_section": _derniere_correspondance_avant(LIVRE_PATTERN, texte_propre, m.start()),
            "titre_section": _derniere_correspondance_avant(TITRE_PATTERN, texte_propre, m.start()),
            "chapitre_section": _derniere_correspondance_avant(CHAPITRE_PATTERN, texte_propre, m.start()),
            "texte": texte_article,
        })

    return articles


# 4. ORCHESTRATION


def main():
    DOSSIER_PDF.mkdir(exist_ok=True)
    tous_les_articles = []
    reussies, echouees = [], []

    # Chaque source est traitée indépendamment : avant ce correctif, une
    # exception sur UNE SEULE source (ex. lien mort, PDF illisible) faisait
    # planter tout le script AVANT l'écriture de corpus_ohada.json — donc
    # même les sources déjà traitées avec succès étaient perdues, et
    # l'ancien fichier restait inchangé sans qu'on sache pourquoi. C'est ce
    # qui explique qu'un Acte ajouté à SOURCES puisse rester absent du
    # corpus sans aucun message d'erreur visible.
    for source in SOURCES:
        try:
            chemin_pdf = telecharger_pdf(source)
            time.sleep(1)  # pause polie entre deux téléchargements

            print(f"[parse] {source['id']}")
            texte_brut = extraire_texte_pdf(chemin_pdf)
            articles = parse_articles(texte_brut, source["id"], source["titre"])

            if not articles:
                # parse_articles affiche déjà un avertissement, mais on la
                # traite ici comme un échec pour qu'elle apparaisse dans le
                # résumé final plutôt que de passer inaperçue.
                raise ValueError("0 article extrait — regex probablement inadaptée à ce PDF")

            print(f"  -> {len(articles)} articles extraits")
            tous_les_articles.extend(articles)
            reussies.append(source["id"])

        except Exception as erreur:
            print(f"[ÉCHEC] {source['id']} : {erreur}")
            echouees.append((source["id"], str(erreur)))
            continue  # on passe à la source suivante au lieu de tout arrêter

    FICHIER_SORTIE.write_text(
        json.dumps(tous_les_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[terminé] {len(tous_les_articles)} articles au total -> {FICHIER_SORTIE.name}")
    print(f"  Sources réussies ({len(reussies)}) : {', '.join(reussies) or '-'}")
    if echouees:
        print(f"  Sources ÉCHOUÉES ({len(echouees)}) :")
        for source_id, erreur in echouees:
            print(f"    - {source_id} : {erreur}")
        print("  -> corpus_ohada.json a quand même été écrit avec les sources "
              "réussies. Corrige les sources en échec et relance le script "
              "(les PDF déjà téléchargés dans pdfs/ ne seront pas retéléchargés).")


if __name__ == "__main__":
    main()
