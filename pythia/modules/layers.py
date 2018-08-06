import torch

from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class GatedTanh(nn.Module):
    '''
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\in R^m => y \in R^n
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    '''
    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = nn.functional.tanh(self.fc(x))
        gated = nn.functional.sigmoid(self.gate_fc(x))

        # Element wise multiplication
        y = y_tilda * gated

        return y


class ClassifierLayer(nn.Module):
    def __init__(self, classifier_type, in_dim, out_dim, **kwargs):
        super(ClassifierLayer, self).__init__()

        if classifier_type == "weight_norm":
            self.module = WeightNormClassifier(in_dim, out_dim, kwargs)
        elif classifier_type == "logit":
            self.module = LogitClassifier(in_dim, out_dim, kwargs)
        elif classifier_type == "linear":
            self.module = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError("Unknown classifier type: %s"
                                      % classifier_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LogitClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(LogitClassifier, self).__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        txt_nonLinear_dim = kwargs['txt_hidden_dim']
        image_nonLinear_dim = kwargs['img_hidden_dim']

        self.f_o_text = GatedTanh(input_dim, txt_nonLinear_dim)
        self.f_o_image = GatedTanh(input_dim, image_nonLinear_dim)
        self.linear_text = nn.Linear(txt_nonLinear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_nonLinear_dim, num_ans_candidates)

        if 'pretrained_image' in kwargs and \
                kwargs['pretrained_text'] is not None:
            self.linear_text.weight.data.copy_(
                torch.from_numpy(kwargs['pretrained_text']))

        if 'pretrained_image' in kwargs and \
                kwargs['pretrained_image'] is not None:
            self.linear_image.weight.data.copy_(
                torch.from_numpy(kwargs['pretrained_image']))

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super(WeightNormClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModalCombineLayer(nn.Module):
    def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
        if combine_type == "MFH":
            self.module = MFH(img_feat_dim, txt_emb_dim, kwargs)
        elif combine_type == "gated_element_multiply":
            self.module = GatedElementMultiply(img_feat_dim, txt_emb_dim,
                                               kwargs)
        elif combine_type == "two_layer_element_multiply":
            self.module = TwoLayerElementMultiply(img_feat_dim, txt_emb_dim,
                                                  kwargs)
        else:
            raise NotImplementedError("Not implemented combine type: %s"
                                      % combine_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MfbExpand(nn.Module):
    def __init__(self, img_feat_dim, txt_emb_dim, hidden_dim, dropout):
        super(MfbExpand, self).__init__()
        self.lc_image = nn.Linear(
            in_features=img_feat_dim,
            out_features=hidden_dim)
        self.lc_ques = nn.Linear(
            in_features=txt_emb_dim,
            out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = (
                torch.unsqueeze(ques1, 1).expand(-1, num_location, -1))
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class MFH(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(MFH, self).__init__()
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.relu = nn.ReLU()

        hidden_sizes = kwargs['hidden_sizes']
        self.out_dim = int(sum(hidden_sizes) / kwargs['pool_size'])

        self.order = kwargs['order']
        self.pool_size = kwargs['pool_size']

        for i in range(self.order):
            mfb_exp_i = MfbExpand(img_feat_dim=image_feat_dim,
                                  txt_emb_dim=ques_emb_dim,
                                  hidden_dim=hidden_sizes[i],
                                  dropout=kwargs['dropout'])
            self.mfb_expand_list.append(mfb_exp_i)
            self.mfb_sqz_list.append(self.mfb_squeeze)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1

        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)

        # append at last feature
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature

    def mfb_squeeze(self, joint_feature):
        # joint_feature dim: N x k x dim or N x dim

        orig_feature_size = len(joint_feature.size())

        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)

        batch_size, num_loc, dim = joint_feature.size()

        if dim % self.pool_size != 0:
            exit("the dim %d is not multiply of \
             pool_size %d" % (dim, self.pool_size))

        joint_feature_reshape = joint_feature.view(
            batch_size, num_loc, int(dim / self.pool_size), self.pool_size)

        # N x 100 x 1000 x 1
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)

        iatt_iq_sqrt = (torch.sqrt(self.relu(iatt_iq_sumpool))
                        - torch.sqrt(self.relu(-iatt_iq_sumpool)))

        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
        iatt_iq_l2 = nn.functional.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(
            batch_size, num_loc, int(dim / self.pool_size))

        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

        return iatt_iq_l2


# need to handle two situations,
# first: image (N, K, i_dim), question (N, q_dim);
# second: image (N, i_dim), question (N, q_dim);
class GatedElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(GatedElementMultiply, self).__init__()
        self.fa_image = GatedTanh(image_feat_dim, kwargs['hidden_size'])
        self.fa_txt = GatedTanh(ques_emb_dim, kwargs['hidden_size'])
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.out_dim = kwargs['hidden_size']

    def forward(self, image_feat, question_embedding):
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)

        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            question_fa_expand = torch.unsqueeze(
                question_fa, 1).expand(-1, num_location, -1)
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TwoLayerElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(TwoLayerElementMultiply, self).__init__()

        self.fa_image1 = GatedTanh(image_feat_dim, kwargs['hidden_size'])
        self.fa_image2 = GatedTanh(
            kwargs['hidden_size'], kwargs['hidden_size'])
        self.fa_txt1 = GatedTanh(ques_emb_dim, kwargs['hidden_size'])
        self.fa_txt2 = GatedTanh(
            kwargs['hidden_size'], kwargs['hidden_size'])

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.out_dim = kwargs['hidden_size']

    def forward(self, image_feat, question_embedding):
        image_fa = self.fa_image2(self.fa_image1(image_feat))
        question_fa = self.fa_txt2(self.fa_txt1(question_embedding))

        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            question_fa_expand = torch.unsqueeze(
                question_fa, 1).expand(-1, num_location, -1)
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class TransformLayer(nn.Module):
    def __init__(self, transform_type, in_dim, out_dim, hidden_dim):
        super(TransformLayer, self).__init__()

        if transform_type == "linear":
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == "conv":
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError(
                "Unknown post combine transform type: %s" % transform_type
            )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim,
                      out_features=out_dim), dim=None)
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=hidden_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_dim,
                               kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

        return iatt_conv3
