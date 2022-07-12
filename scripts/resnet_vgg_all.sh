python main_all.py --model resnet --shape 0 --gpu-id 0 &
python main_all.py --model resnet --shape 1 --gpu-id 1 &
python main_all.py --model resnet --shape 2 --gpu-id 3 &
python main_all.py --model resnet --shape 3 --gpu-id 4
python main_all.py --model vgg  --shape 0 --gpu-id 0 &
python main_all.py --model vgg  --shape 1 --gpu-id 1 &
python main_all.py --model vgg  --shape 2 --gpu-id 3 &
python main_all.py --model vgg  --shape 3 --gpu-id 4
python main_all.py --model inception  --shape 0 --gpu-id 0 &
python main_all.py --model inception  --shape 1 --gpu-id 1 &
python main_all.py --model inception  --shape 2 --gpu-id 3 &
python main_all.py --model inception  --shape 3 --gpu-id 4
