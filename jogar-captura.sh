#!/bin/sh
# Como o jogar.sh, mas com a captura dos passes de desenho armada.
# Quando a cena com defeito estiver na tela (de preferência parado), em outro terminal:
#     touch work/captura/gatilho
# O motor captura os passes do quadro seguinte em work/captura (cada passe em PPM + passes.txt
# + draws.txt). Fica lento por alguns segundos; depois volta ao normal. Um gatilho por sessão.
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/work/captura"
rm -f "$here/work/captura/"* 2>/dev/null
export RSX_ENGINE_PASS_DUMP="$here/work/captura"
export RSX_ENGINE_PASS_DUMP_TRIGGER="$here/work/captura/gatilho"
export RSX_ENGINE_PASS_DUMP_DRAWS=1
export RSX_ENGINE_PASS_DUMP_MAX=${CAPTURA_MAX:-160}
exec "$here/jogar.sh" "$@"
