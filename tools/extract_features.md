## Extract Image Features

We use detectron to extract image features. Set up [detectron](https://github.com/facebookresearch/Detectron) 
and copy [tools/extract_features.py](tools/extract_features.py) to detectron for extracting features


Feature extraction works best with commit #3a38b7b of [detectron](https://github.com/facebookresearch/Detectron)
and #0dd3284 of [caffe2](https://github.com/caffe2/caffe2)


download the pretrained detectron model
```bash
wget https://dl.fbaipublicfiles.com/pythia/detectron_model/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl
wget https://dl.fbaipublicfiles.com/pythia/detectron_model/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml

$INPUT_DIR = /path/to/your/input/image or directory

python extract_features.py --cfg e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml \
--wts FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl \
--min_bboxes 100 --max_bboxes 100 \
--feat_name gpu_0/fc6 \
--output_dir ~/temp_out $INPUT_DIR
