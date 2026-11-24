#!/usr/bin/env bash
# Instala o AgroClima numa VM Ubuntu (Oracle Cloud Free Tier).
# Uso, dentro da VM:  bash setup.sh https://github.com/feliperuizz/agroglima.git
set -euo pipefail

REPO="${1:-https://github.com/feliperuizz/agroglima.git}"
APP_DIR="$HOME/agroclima"

echo "== 0/5 Memória extra (swap de 2 GB, importante na VM Micro de 1 GB)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

echo "== 1/5 Pacotes do sistema"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip nginx git iptables-persistent

echo "== 2/5 Código"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO" "$APP_DIR"
fi

echo "== 3/5 Ambiente Python"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "== 4/5 Serviço (systemd) e proxy (nginx)"
sed "s|__USER__|$USER|g; s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/deploy/agroclima.service" \
  | sudo tee /etc/systemd/system/agroclima.service > /dev/null
sudo cp "$APP_DIR/deploy/nginx-agroclima.conf" /etc/nginx/sites-available/agroclima
sudo ln -sf /etc/nginx/sites-available/agroclima /etc/nginx/sites-enabled/agroclima
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now agroclima
sudo systemctl restart nginx

echo "== 5/5 Firewall interno da imagem Oracle (portas 80 e 443)"
sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

echo
echo "Pronto. Teste local:  curl -s http://127.0.0.1/api/stats | head -c 200"
echo "No navegador:        http://$(curl -s ifconfig.me)/"
