# AGENTS — Règles opérationnelles

## Planification
- Toujours décomposer en steps atomiques. Un step = une action.
- Utiliser spider_search pour analyser une URL spécifique (scraping du contenu réel de la page).
- Utiliser web_search pour des recherches générales sur le web (infos récentes, actualités).
- Ne jamais utiliser web_search pour analyser un site — utiliser spider_search avec l'URL directement.
- Utiliser code pour valider les calculs plutôt que calculer de tête.
- Préférer file pour lire/écrire des fichiers locaux plutôt que de deviner leur contenu.

## Qualité
- Répondre dans la langue de l'utilisateur.
- Écrire les accents correctement (é, è, ê, à, ç, ù, ô).
- Ne jamais tronquer une réponse — aller jusqu'au bout.

## Sécurité
- Ne jamais exécuter de code qui supprime des fichiers sans confirmation.
- Ne jamais afficher de tokens, mots de passe ou clés API dans les réponses.
- Limiter les appels web_search à 5 par plan pour éviter le rate limiting.
