---
id: metrics
title: Adding a custom metric
sidebar_label: Adding a custom metric
---

# Custom Metrics

This is a tutorial on how to add a new metric to MMF.

MMF is agnostic to the kind of metrics that can be added to it.
Adding a metric requires adding a metric class and adding your new metric to your config yaml.
For example, the [ConcatBERT](https://github.com/facebookresearch/mmf/blob/main/website/docs/tutorials/concat_bert_tutorial.md) model uses the `binary_f1` metric when evaluating on the hateful memes dataset.
The metric class is `BinaryF1` defined in [mmf/modules/metrics.py](https://github.com/facebookresearch/mmf/blob/main/mmf/modules/metrics.py)
The metric key `binary_f1` is added to the list of metrics in the config yaml at [mmf/projects/hateful_memes/configs/concat_bert/defaults.yaml](https://github.com/facebookresearch/mmf/blob/15fa63071bfaed56db43deba871cfec76439c66f/projects/others/concat_bert/hateful_memes/defaults.yaml#L28).


# Metric Class

Add your metric class to metrics.py. It should be a subclass of `BaseMetric`.
Metrics should implement a function calculate with signature `calculate(self, sample_list, model_output, *args, **kwargs)`,
where sample_list (`SampleList`) is the current batch and model_output is a dict return by your model for current sample_list.

```python
@registry.register_metric("f1")
class F1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("f1")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = f1_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)

@registry.register_metric("binary_f1")
class BinaryF1(F1):
    """Metric for calculating Binary F1.

    **Key:** ``binary_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_f1"
```

# Metrics Config

Add the name of your new metric class to your evaluation config. Multiple metrics can be specified using a yaml array.

```yaml
evaluation:
  metrics:
  - accuracy
  - roc_auc
  - binary_f1
```

For metrics that take parameters your yaml config will specify params. You can also specify a custom key to be assigned to the metric. For [example](https://github.com/facebookresearch/mmf/blob/main/projects/unit/configs/vg/single_task.yaml),

```yaml
evaluation:
  metrics:
  - type: detection_mean_ap
    key: detection_mean_ap
    datasets:
    - detection_visual_genome
    params:
      dataset_json_files:
        detection_visual_genome:
          val: ${env.data_dir}/datasets/visual_genome/detection_split_by_coco_2017/annotations/instances_val_split_by_coco_2017.json

```

If your model uses early stopping, make sure that the early_stop.criteria is added as an evaluation metric. For example the [vizwiz](https://github.com/facebookresearch/mmf/blob/main/projects/ban/configs/vizwiz/defaults.yaml) config,

```yaml
evaluation:
  metrics:
  - vqa_accuracy

training:
  early_stop:
    criteria: vizwiz/vqa_accuracy
    minimize: false
```

# Multi-Metric Classes

If a loss class is responsible for calculating multiple metrics, for example, maybe due to shared calculations, you can return a dictionary of tensors.

For an example, take a look at the `BinaryF1PrecisionRecall` class in [mmf/modules/metrics.py](https://github.com/facebookresearch/mmf/blob/main/mmf/modules/metrics.py)

```python
@registry.register_metric("f1_precision_recall")
class F1PrecisionRecall(BaseMetric):
    """Metric for calculating F1 precision and recall.
    params will be directly passed to sklearn
    precision_recall_fscore_support function.
    **Key:** ``f1_precision_recall``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("f1_precision_recall")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1_precision_recall and return it back as a dict.
        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.
        Returns:
            Dict(
                'f1':         torch.FloatTensor,
                'precision':  torch.FloatTensor,
                'recall':     torch.FloatTensor
            )
        """
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value_tuple = precision_recall_fscore_support(
            expected.cpu(), output.cpu(), **self._sk_kwargs
        )
        value = {
            "precision": expected.new_tensor(value_tuple[0], dtype=torch.float),
            "recall": expected.new_tensor(value_tuple[1], dtype=torch.float),
            "f1": expected.new_tensor(value_tuple[2], dtype=torch.float),
        }
        return value


@registry.register_metric("binary_f1_precision_recall")
class BinaryF1PrecisionRecall(F1PrecisionRecall):
    """Metric for calculating Binary F1 Precision and Recall.
    **Key:** ``binary_f1_precision_recall``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_f1_precision_recall"
```
