# Converstion of GQA to VQA format

* Download GQA datasets and store as format shown in conversion script
* Download glove embeddings 300D file
* Run the script from the root of the repo as by changing relevant paths:

```
python tools/scripts/gqa/convert_gqa_to_vqa.py --gqa_dir <path_to_gqa_dir> --out_dir <path_to_out_dir>
```
