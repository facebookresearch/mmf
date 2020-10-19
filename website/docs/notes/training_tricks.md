---
id: training_tricks
title: Training Tricks
sidebar_label: Training Tricks
---

# FP16

MMF supports FP16 training for faster performance with negligible impact on
the results through `torch.cuda.amp` module. Append `training.fp16=True` to
the end of your command to enable fp16 training in the trainer.
