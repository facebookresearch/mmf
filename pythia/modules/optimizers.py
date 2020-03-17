from pythia.common.registry import registry
from transformers.optimization import AdamW


registry.register_optimizer("adam_w")(AdamW)
