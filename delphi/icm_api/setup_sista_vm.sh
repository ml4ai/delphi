# Script to setup AutoMATES and Delphi webapps on SISTA VM

cd
sudo apt-get install vim
git clone https://github.com/ml4ai/delphi
git clone https://github.com/ml4ai/automates
git clone https://github.com/sorgerlab/indra
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh
source ~/.bashrc
conda create -n automates
conda activate automates
sudo apt-get install graphviz libgraphviz-dev pkg-config apache2 apache2-dev default-jre

cd delphi
pip install -e .
cd ~/indra
pip install -e .
cd

cd automates/demo
pip install -r requirements.txt
sudo ln -sfT ~/automates/demo /var/www/html/automates_demo
sudo ln -sfT ~/delphi /var/www/html/delphi
wget http://vision.cs.arizona.edu/adarsh/delphi.db
sudo ln -sfT ~/delphi.db /var/www/html/delphi.db
