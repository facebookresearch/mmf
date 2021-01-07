---
id: model_zoo
title: Model Zoo
sidebar_label: Model Zoo
---

Here is the list of models currently implemented in MMF:

| Model           | Key                                  | Datasets                                 | Notes
| --------------- | ------------------------------------ | ---------------------------------------- | ----------------------------------------------------------- |
| BAN             | ban                                  | textvqa, vizwiz, vqa2                    | [BAN](https://arxiv.org/abs/1805.07932) support is preliminary and hasn't been properly fine-tuned yet. |
| BUTD            | butd                                 | coco                                     | [paper](https://arxiv.org/abs/1707.07998)                   |
| CNN LSTM        | cnn_lstm                             | clevr                                    |                                                             |
| FUSIONS         | concat_bert, late_fusion, concat_bow | hateful_memes                            |                                                             |
| LoRRA           | lorra                                | vqa2, textvqa, vizwiz                    | [paper](https://arxiv.org/abs/1904.08920)                   |
| LXMERT          | lxmert                               | coco, gqa, visual_genome, vqa2           | [paper](https://arxiv.org/abs/1908.07490)                   |
| M4C             | m4c                                  | ocrvqa, stvqa, textvqa                   | [paper](https://arxiv.org/pdf/1911.06258.pdf)               |
| M4C Captioner   | m4c_captioner                        | coco, textcaps                           | [paper](https://arxiv.org/pdf/2003.12462.pdf)               |
| MMBT            | mmbt                                 | hateful_memes, coco, mmimdb, okvqa, vqa2 | [paper](https://arxiv.org/abs/1909.02950)                   |
| MMF Transformer | mmf_transformer                      | hateful_memes, okvqa, vqa2               |                                                             |
| Movie MCAN      | movie_mcan                           | vqa2                                     | [paper](https://arxiv.org/abs/2004.11883)                   |
| Pythia          | pythia                               | textvqa, vizwiz, vqa2, visual_genome     | [paper](https://arxiv.org/abs/1904.08920)                   |
| Unimodal        | unimodal                             | hateful_memes                            |                                                             |
| VilBERT         | vilbert                              | hateful_memes, coco, conceptual_captions, vqa2, mmimdb, nlvr2, visual_entailment, vizwiz, vqa2 |[paper](https://arxiv.org/abs/1908.02265)|
| Visual BERT     | visual_bert                          | gqa, hateful_memes, localized_narratives, coco, conceptual_captions, sbu, vqa2, mmimdb, nlvr2, visual_entailment, vizwiz|[paper](https://arxiv.org/abs/1908.03557)|

We are adding many more new models which will be available soon.
