config="--config config/style_transfer/pama.yml"

for i in $(seq 1 140)   
do   
python pama_inference.py $config --content data/golden_gate.jpg --style data/styles/$i.jpg
python pama_inference.py $config --content data/lego/train/r_17.png --style data/styles/$i.jpg
python pama_inference.py $config --content data/nerf_llff_data/fern/images/IMG_4026.JPG --style data/styles/$i.jpg
python pama_inference.py $config --content data/nerf_llff_data/flower/images/IMG_2962.JPG --style data/styles/$i.jpg
python pama_inference.py $config --content data/nerf_llff_data/leaves/images/IMG_2997.JPG --style data/styles/$i.jpg
python pama_inference.py $config --content data/nerf_llff_data/orchids/images/IMG_4467.JPG --style data/styles/$i.jpg
python pama_inference.py $config --content data/nerf_llff_data/trex/images/DJI_20200223_163548_810.jpg --style data/styles/$i.jpg
python pama_inference.py $config --content data/NSVF/Synthetic_NSVF/Palace/rgb/0_train_0000.png --style data/styles/$i.jpg
done
