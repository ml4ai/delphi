# Script to setup AutoMATES and Delphi webapps on SISTA VM
# This script has not been adequately tested - run at your own risk!
# (Perhaps try running line by line if you haven't run the whole thing before)
# TODO Convert this into a README instead - too dangerous to let it live as a
# script.
# TODO Add proper instructions about chowning etc.

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
sudo mkdir /var/www/html/delphi_data
cd /var/www/html/delphi_data
wget http://vision.cs.arizona.edu/adarsh/delphi.db
cd
sudo chown -R www-data:www-data /var/www/html/delphi_data
