#!/bin/sh
# Rende l'anteprima per i link: 1200x630 (Space, X, LinkedIn, Facebook) e 1280x640 (anteprima social di GitHub).
# Serve Google Chrome; i font sono quelli della demo in web/fonts/.
set -e
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
for size in 1200,630 1280,640; do
  w=${size%,*}; h=${size#*,}
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
    --allow-file-access-from-files --virtual-time-budget=2000 \
    --window-size="$w,$h" --screenshot="$PWD/social_card_${w}x${h}.png" "file://$PWD/social_card.html" 2>/dev/null
done
ls -la social_card_*.png
