import torch

from torch import nn

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ConvNet, Flatten


@registry.register_model("cnn_lstm")
class CNNLSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    def build(self):
        num_question_choices = registry.get(self._datasets[0] + "_text_vocab_size")
        num_answer_choices = registry.get(self._datasets[0] + "_num_final_outputs")

        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )
        self.lstm = nn.LSTM(**self.config.lstm)

        layers_config = self.config.cnn.layers
        conv_layers = []
        for i in range(len(layers_config.input_dims)):
            conv_layers.append(
                ConvNet(
                    layers_config.input_dims[i],
                    layers_config.output_dims[i],
                    kernel_size=layers_config.kernel_sizes[i]
                )
            )
        conv_layers.append(Flatten())
        self.cnn = nn.Sequential(*conv_layers)

        self.classifier = nn.Linear(self.config.classifier.input_dim, num_answer_choices)

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        question = sample_list.text
        image = sample_list.image

        _, hidden = self.lstm(self.text_embedding(question))
        hidden = hidden[0].transpose(0, 1)
        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)

        image = self.cnn(image)
        fused = torch.cat([hidden, image], dim=-1)
        scores = self.classifier(fused)


        return {"scores": scores}
