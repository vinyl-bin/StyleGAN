apt-get update
apt-get --fix -missing
apt-get install git
git clone https://github.com/justinpinkney/stylegan2
cd stylegan2
nvcc test_nvcc.cu -o test_nvcc -run

mkdir raw
mkdir aligned
mkdir generated

apt-get install wget
apt-get update
wget https://www.bntnews.co.kr/data/bnt/image/201008/cba489d1b74edb358cfcc9fe5c84d23c.jpg -O raw/example.jpg
