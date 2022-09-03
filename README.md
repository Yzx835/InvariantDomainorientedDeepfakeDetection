# code of Improving Deepfake Detection Generalization by Invariant Risk Minimization

## Requirements
```
albumentations==1.0.3
dlib==19.15.0
facenet-pytorch==2.5.2
opencv-python==4.5.3.56
scikit-image==0.18.3
scipy==1.4.1
torch==1.8.2
torchaudio==0.8.2
torchvision==0.9.2
```
## Usage
### Dataset
The organization of faceforensics++ is as follows. The faceforensics++ dataset consists of raw, deepfakes, faceswaps, face2face, and neuraltextures. The **deepfakes_swap** and **faceswap_swap** directories contain the self-face-swapping faces, which are generated by the identity swap method like deepfake and faceswap relatively. While the face2face and neuraltextures generation methods are based on face reenactment.
<pre>
data
├── deepfakes
├── deepfakes_swap
├── face2face
├── faceswap
├── faceswap_swap
├── neuraltextures
└── raw
</pre>
### Train
```
# train xception on ff++ dataset with DRDA strategy and DABN moudle.
# the region_erase_pathx is the path of a json file which contains the point of region to erase.
python train.py --datapath='./data' --splitpath='./splitpath' \
    --region_erase_path1='./region_erase_path1' --region_erase_path2='./region_erase_path2' --epoch=15 --batchsize=48 --DABN=3
```
### Test
```
# test xception on ff++ with DABN
python test.py --datapath='./data' --splitpath='./splitpath' \
    --modelpath='./modelpath'
```