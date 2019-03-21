# Script to setup AutoMATES demo on SISTA VM

cd
sudo apt-get install vim
git clone https://github.com/ml4ai/delphi
git clone https://github.com/ml4ai/automates
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh
source ~/.bashrc
conda create -n automates
conda activate automates
sudo apt-get install graphviz libgraphviz-dev pkg-config apache2 apache2-dev default-jre
cd automates/demo
pip install -r requirements.txt
sudo ln -sfT ~/automates/demo /var/www/html/automates_demo
sudo ln -sfT ~/delphi /var/www/html/delphi
sudo mkdir -p /tmp/automates
sudo chown -R www-data /tmp/automates
