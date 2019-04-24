# Terminology and Concepts

**Authors**: Amanpreet Singh

To develop on top of Pythia, it is necessary to understand concepts and terminology
used in Pythia codebase. Pythia has been very carefully designed from ground-up to be a
multi-tasking framework. This means using Pythia you can train on multiple tasks/datasets
together.

To achieve this, Pythia has few opinions about architecture of your research project.
But, being generic means Pythia abstracts a lot of concepts in its modules and it would
be easy to develop on top of Pythia once a developer understands these simple concepts.
Major concepts and terminology in Pythia that one needs to know in order to develop
over Pythia are as follows:

- [Tasks and Datasets](#tasks-and-datasets)
- [Models](#models)
- [Registry](#registry)
- [Configuration](#configuration)
- [Processors](#processors)
- [Sample List](#sample-list)


## Tasks and Datasets

In Pythia, we have divided datasets into a set category of tasks. Thus, a task corresponds
to a collection of datasets that belong to it. For example, VQA 2.0, VizWiz and TextVQA
all belong VQA task. Each task and dataset has been assigned a unique key which is used
to refer it in the command line arguments.

Following table shows the tasks and their datasets:

```eval_rst
+--------+------------+------------------------+
|**Task**| **Key**    | **Datasets**           |
+--------+------------+------------------------+
| VQA    | vqa        | VQA2.0, VizWiz, TextVQA|
+--------+------------+------------------------+
| Dialog | dialog     | VisualDialog           |
+--------+------------+------------------------+
```

Following table shows the inverse of the above table, datasets along with their tasks and keys:

```eval_rst
+--------------+---------+-----------+--------------------+
| **Datasets** | **Key** | **Task**  |**Notes**           |
+--------------+---------+-----------+--------------------+
| VQA 2.0      | vqa2    | vqa       |                    |
+--------------+---------+-----------+--------------------+
| TextVQA      | textvqa | vqa       |                    |
+--------------+---------+-----------+--------------------+
| VizWiz       | vizwiz  | vqa       |                    |
+--------------+---------+-----------+--------------------+
| VisualDialog | visdial | dialog    |   Coming soon!     |
+--------------+---------+-----------+--------------------+
```

## Models

Reference implementations for state-of-the-art models have been included to act as
a base for reproduction of research papers and starting point of new research. Pythia has
been used in past for following papers:

- [Towards VQA Models That Can Read (LoRRA model)](https://arxiv.org/abs/1904.08920)
- [VQA 2018 Challenge winner](https://arxiv.org/abs/1807.09956)
- VizWiz 2018 Challenge winner

Similar to tasks and datasets, each model has been registered with a unique key for easy
reference in configuration and command line arguments. Following table shows each model's
key name and datasets it can be run on.

```eval_rst
+-----------+---------+-----------------------+
| **Model** | **Key** | **Datasets**          |
+-----------+---------+-----------------------+
| LoRRA     | lorra   | textvqa, vizwiz       |
+-----------+---------+-----------------------+
| Pythia    | pythia  | textvqa, vizwiz, vqa2 |
+-----------+---------+-----------------------+
| BAN       | ban     | textvqa, vizwiz, vqa2 |
+-----------+---------+-----------------------+
```
**Note**: BAN support is preliminary and hasn't been properly fine-tuned yet.

## Registry

Registry acts as a central source of truth for Pythia. Inspired from Redux's global store,
useful information needed by Pythia ecosystem is registered in the `registry`. Registry can be
considered as a general purpose storage for information which is needed by multiple parts
of the framework and acts source of information wherever that information is needed.

Registry also registers models, tasks, datasets etc. based on a unique key as mentioned above.
Registry's functions can be used as decorators over the classes which need to be registered
(for e.g. models etc.)

Registry object can be imported as the follow:

```
from pythia.common.registry import registry
```

Find more details about Registry class in its documentation [common/registry](../common/registry).
