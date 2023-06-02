cd /home/toonify/stylegan2
python align_images.py raw aligned
python project_images.py --num-steps 500 aligned generated
