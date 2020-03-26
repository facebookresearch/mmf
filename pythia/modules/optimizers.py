from pythia.common.registry import registry
from pytorch_transformers.optimization import AdamW


registry.register_optimizer("adam_w")(AdamW)
