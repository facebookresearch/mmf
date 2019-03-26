# Converstion of GQA to VQA format

* Download GQA datasets and store as format shown in conversion script
* Download glove embeddings 300D file
* Run the script from the root of the repo as by changing relevant paths: 

```
PYTHONPATH=. python ./data_prep/gqa/convert_gqa_to_vqa.py \
             --gqa_dir /checkpoint/meetshah/datasets/gqa/ \
             --out_dir /checkpoint/meetshah/datasets/gqa_pp/
```
