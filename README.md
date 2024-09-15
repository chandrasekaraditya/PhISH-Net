Python 3.8

download data to ./data in the same format. create a folder splits with all_list.txt, train_list.txt and test_list.txt

conda create -n pnet python=3.8
conda activate pnet
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install opencv-python scikit-image tqdm rawpy Pillow==9.3.0 h5py==3.9.0 imageio==2.19.3 scikit-image==0.19.3
