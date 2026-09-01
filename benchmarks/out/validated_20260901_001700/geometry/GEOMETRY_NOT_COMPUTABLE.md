# Geometry evaluation status

The conservative `geometry_eval.py --match unique` run was attempted for the saved repeat-01/base layers of Gemma4 12B, GPT-OSS 20B and Gemma3 12B.

All three attempts stopped at the same GEOS topology error (`side location conflict` near `345211.3551, 6652605.9703`). No geometry score CSV is valid or available.

Additional coverage limits: only 12/95 expert rows have an unambiguous public reference match; 45 private links require `YANDEX_OAUTH_TOKEN`; `territories.json` is unavailable, so no project-territory clipping was performed.
