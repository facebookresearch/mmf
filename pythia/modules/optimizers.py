from transformers.optimization import AdamW

from pythia.common.registry import registry

registry.register_optimizer("adam_w")(AdamW)
