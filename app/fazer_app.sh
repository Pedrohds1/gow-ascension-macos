#!/bin/sh
# Monta "God of War Ascension.app" na pasta do projeto: o executável recompilado vai para dentro do
# bundle (Contents/MacOS/MyGameRecomp), e o lançador chama o jogar.sh do projeto com EXE apontando
# para ele (ISO, saves, catálogo de shaders e cache de SPIR-V continuam na pasta do projeto).
# Refazer depois de cada build:  app/fazer_app.sh   (só com o jogo fechado).
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
app="$here/God of War Ascension.app"
exe="$here/build/gow-mac-vk-nat/MyGameRecomp"
[ -x "$exe" ] || { echo "falta o executável: $exe"; exit 1; }
if pgrep -x MyGameRecomp >/dev/null; then echo "feche o jogo antes de refazer o app"; exit 1; fi

tmp="$app.tmp"; rm -rf "${tmp:?}"
mkdir -p "$tmp/Contents/MacOS" "$tmp/Contents/Resources"
cp "$exe" "$tmp/Contents/MacOS/MyGameRecomp"

# Ícone: das imagens do próprio disco (montado em work/disc por jogar.sh).
if [ ! -f "$here/work/icone-1024.png" ]; then
    [ -d "$here/work/disc/PS3_GAME" ] || { echo "monte a ISO (rode ./jogar.sh uma vez)"; exit 1; }
    "$here/.venv/bin/python" "$here/app/fazer_icone.py" "$here/work/disc/PS3_GAME" "$here/work/icone-1024.png"
fi
set_dir="$here/work/AppIcon.iconset"; rm -rf "${set_dir:?}"; mkdir -p "$set_dir"
for s in 16 32 128 256 512; do
    sips -z $s $s "$here/work/icone-1024.png" --out "$set_dir/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2)); sips -z $d $d "$here/work/icone-1024.png" --out "$set_dir/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$set_dir" -o "$tmp/Contents/Resources/AppIcon.icns"
rm -rf "${set_dir:?}"

cat > "$tmp/Contents/MacOS/GodOfWarAscension" <<SH
#!/bin/sh
# Lançador: o jogo roda a partir da pasta do projeto (ISO, saves, caches).
bundle=\$(cd "\$(dirname "\$0")/../.." && pwd)
exec env EXE="\$bundle/Contents/MacOS/MyGameRecomp" "$here/jogar.sh" "\$@"
SH
chmod +x "$tmp/Contents/MacOS/GodOfWarAscension"

cat > "$tmp/Contents/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>God of War Ascension</string>
    <key>CFBundleDisplayName</key><string>God of War: Ascension</string>
    <key>CFBundleIdentifier</key><string>local.pedro.gow-ascension-mac</string>
    <key>CFBundleExecutable</key><string>GodOfWarAscension</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>$(date +%Y%m%d%H%M)</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.action-games</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>GCSupportsControllerUserInteraction</key><true/>
</dict>
</plist>
PL

codesign --force --deep -s - "$tmp" >/dev/null 2>&1 || true
rm -rf "${app:?}"; mv "$tmp" "$app"
touch "$app"
echo "pronto: $app"
