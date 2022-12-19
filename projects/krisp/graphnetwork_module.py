# Copyright (c) Facebook, Inc. and its affiliates.

# This is the generic module for Graph Network computations,
# which can be added as a component to any base network
# Used some word2vec code from https://github.com/adithyamurali/TaskGrasp
# Also used example code from https://github.com/rusty1s/pytorch_geometric
import os
import pickle

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.text import VocabDict
from networkx import convert_node_labels_to_integers
from torch_geometric.nn import BatchNorm, GCNConv, RGCNConv, SAGEConv
from tqdm import tqdm


def k_hop_subgraph(
    node_idx,
    num_hops,
    edge_index,
    relabel_nodes=False,
    num_nodes=None,
    flow="source_to_target",
):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def make_graph(
    raw_graph,
    prune_culdesacs=False,
    prune_unconnected=True,
    q_vocab=None,
    i_vocab=None,
    ans_vocab=None,
    include_reverse_relations=False,
):
    # None edge cases
    if q_vocab is None:
        q_vocab = []
    q_vocab = set(q_vocab)
    if i_vocab is None:
        i_vocab = []
    i_vocab = set(i_vocab)
    if ans_vocab is None:
        ans_vocab = []
    ans_vocab = set(ans_vocab)

    # Init nx graph
    graph = nx.DiGraph()

    # Add the nodes
    for concept in raw_graph["concepts2idx"]:
        # Add node type info
        q_node = concept in q_vocab
        i_node = concept in i_vocab
        ans_node = concept in ans_vocab

        # Add node
        graph.add_node(concept, q_node=q_node, i_node=i_node, ans_node=ans_node)

    # Go through edges in raw_graph
    for triplet in raw_graph["triplets"]:
        # Get triplet
        head_idx = triplet[0]
        rel_idx = triplet[1]
        tail_idx = triplet[2]

        # Get the names
        head = raw_graph["concepts"][head_idx]
        tail = raw_graph["concepts"][tail_idx]
        rel = raw_graph["relations"][rel_idx]

        # Add to the graph
        assert head in graph.nodes and tail in graph.nodes
        graph.add_edge(head, tail, relation=rel)

    # Prune totally unconnected nodes
    if prune_unconnected:
        for concept in raw_graph["concepts2idx"]:
            assert concept in graph.nodes

            # Get edges to/from that node
            connecting_edges = list(graph.in_edges(concept)) + list(
                graph.out_edges(concept)
            )

            # Remove if there are no edges
            if len(connecting_edges) == 0:
                graph.remove_node(concept)

    # Prune graph of nodes
    # Custom concepts to remove
    to_remove = [""]
    for concept in to_remove:
        if concept in graph.nodes:
            graph.remove_node(concept)

    # Get the idx graph for easy conversions
    graph_idx = convert_node_labels_to_integers(graph)

    # Also go ahead and return dicts and edge_type and edge_index
    edge_index, edge_type = get_edge_idx_type(
        graph, graph_idx, raw_graph["relations2idx"], include_reverse_relations
    )

    return graph, graph_idx, edge_index, edge_type


def get_edge_idx_type(graph, graph_idx, rel2idx, include_reverse_relations=False):
    # Return from a graph, the required edge_index and edge_type info
    # Pretty simple since from graph_idx
    edge_index = np.array(list(graph_idx.edges)).T

    # For type, need to do a conversion
    edge_type = [graph.edges[e]["relation"] for e in graph.edges]
    edge_type = np.array([rel2idx[rel] for rel in edge_type])

    # Add reverse relations
    if include_reverse_relations:
        edge_src = np.expand_dims(edge_index[0, :], 0)
        edge_dest = np.expand_dims(edge_index[1, :], 0)
        edge_reverse = np.concatenate([edge_dest, edge_src], axis=0)
        edge_index = np.concatenate([edge_index, edge_reverse], axis=1)

    return edge_index, edge_type


def prepare_embeddings(node_names, embedding_file, add_split):
    """
    This function is used to prepare embeddings for the graph
    :param embedding_file: location of the raw embedding file
    :return:
    """
    print("\n\nCreating node embeddings...")

    embedding_model = ""
    if "glove" in embedding_file:
        embedding_model = "glove"
    elif "GoogleNews" in embedding_file:
        embedding_model = "word2vec"
    elif "subword" in embedding_file:
        embedding_model = "fasttext"
    elif "numberbatch" in embedding_file:
        embedding_model = "numberbatch"

    def transform(compound_word):
        return [
            compound_word,
            "_".join([w.lower() for w in compound_word.split(" ")]),
            "_".join([w.capitalize() for w in compound_word.split(" ")]),
            "-".join([w for w in compound_word.split(" ")]),
            "-".join([w for w in compound_word.split(" ")]),
        ]

    node2vec = {}
    model = None

    # glove has a slightly different format
    if embedding_model == "glove":
        tmp_file = ".".join(embedding_file.split(".")[:-1]) + "_tmp.txt"
        glove2word2vec(embedding_file, tmp_file)
        embedding_file = tmp_file

    # Important: only native word2vec file needs binary flag to be true
    print(f"Loading pretrained embeddings from {embedding_file} ...")
    model = KeyedVectors.load_word2vec_format(
        embedding_file, binary=(embedding_model == "word2vec")
    )

    # retrieve embeddings for graph nodes
    no_match_nodes = []
    match_positions = []

    for node_name in tqdm(node_names, desc="Prepare node embeddings"):
        try_words = []
        try_words.extend(transform(node_name))

        # Try to find w2v
        found_mapping = False
        for i, try_word in enumerate(try_words):
            try:
                node2vec[node_name] = model.get_vector(try_word)
                match_positions.append(i + 1)
                found_mapping = True
            except KeyError:
                pass
            if found_mapping:
                break

        # Try multi-words (average w2v)
        if add_split:
            if not found_mapping and len(node_name.split(" ")) > 1:
                sub_word_vecs = []
                for subword in node_name.split(" "):
                    # Get w2v for the individual words
                    try_words = []
                    try_words.extend(transform(subword))
                    mp = []
                    found_submap = False
                    for i, try_word in enumerate(try_words):
                        try:
                            sub_word_vecs.append(model.get_vector(try_word))
                            mp.append(i + 1)
                            found_submap = True
                        except KeyError:
                            pass
                        if found_submap:
                            break

                # If all subswords successful, add it to node2vec and match_positions
                if len(sub_word_vecs) == len(node_name.split(" ")):
                    node2vec[node_name] = np.mean(sub_word_vecs, 0)
                    match_positions.append(
                        np.mean(mp)
                    )  # I'm sort of ignoring match_positions except for counts
                    found_mapping = True
        else:
            if not found_mapping and len(node_name.split("_")) > 1:
                sub_word_vecs = []
                for subword in node_name.split("_"):
                    # Get w2v for the individual words
                    try_words = []
                    try_words.extend(transform(subword))
                    mp = []
                    found_submap = False
                    for i, try_word in enumerate(try_words):
                        try:
                            sub_word_vecs.append(model.get_vector(try_word))
                            mp.append(i + 1)
                            found_submap = True
                        except KeyError:
                            pass
                        if found_submap:
                            break

                # If all subswords successful, add it to node2vec and match_positions
                if len(sub_word_vecs) == len(node_name.split("_")):
                    node2vec[node_name] = np.mean(sub_word_vecs, 0)
                    match_positions.append(
                        np.mean(mp)
                    )  # I'm sort of ignoring match_positions except for counts
                    found_mapping = True

        # All else fails, it's a no match
        if not found_mapping:
            no_match_nodes.append([node_name, try_words])


# This just wraps GraphNetworkModule for mmf so GNM can be a submodule of
# other networks too
@registry.register_model("graph_network_bare")
class GraphNetworkBare(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/graph_network_bare/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):
        extra_config = {}
        extra_config["feed_vb_to_graph"] = False
        extra_config["feed_q_to_graph"] = False
        extra_config["feed_mode"] = None
        extra_config["feed_graph_to_vb"] = False
        extra_config["feed_special_node"] = False
        extra_config["topk_ans_feed"] = None
        extra_config["compress_crossmodel"] = False
        extra_config["crossmodel_compress_dim"] = None
        extra_config["analysis_mode"] = False
        extra_config["noback_vb"] = False
        self.graph_module = GraphNetworkModule(self.config, extra_config)

        if self.config.output_type in [
            "graph_level",
            "graph_level_ansonly",
            "graph_level_inputonly",
        ]:
            # Make a linear last layer
            self.fc = nn.Linear(self.graph_module.gn.output_dim, self.config.num_labels)
        else:
            assert self.config.output_type in ["graph_prediction"]
            # Output is size of graph

    # Each model in MMF gets a dict called sample_list which contains
    # all of the necessary information returned from the image
    def forward(self, sample_list):
        # Forward with sample list
        output = self.graph_module(sample_list)

        # If graph_level, we need to now predict logits
        if self.config.output_type in [
            "graph_level",
            "graph_level_ansonly",
            "graph_level_inputonly",
        ]:
            # Do last layer
            logits = self.fc(output)
        else:
            assert self.config.output_type in ["graph_prediction"]
            logits = output

        # Do zerobias
        logits -= 6.58

        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # If we're in eval / analysis mode, add more to output
        if self.config.analysis_mode:
            output = self.graph_module.add_analysis_to_output(output)

        # MMF will automatically calculate loss
        return output


# Do indirect path stuff with mmf
def mmf_indirect(path):
    if os.path.exists(path):
        return path
    else:
        path = os.path.join(os.getenv("MMF_DATA_DIR"), "datasets", path)
        return path


# Graph network module
# Can be added as part of a larger network, or used alone using GraphNetworkBare
class GraphNetworkModule(nn.Module):
    """The generic class for graph networks
    Can be generically added to any other kind of network
    """

    def __init__(self, config, config_extra=None):
        super().__init__()
        self.config = config
        if config_extra is None:
            self.config_extra = {}
        else:
            self.config_extra = config_extra

        # Load the input graph
        raw_graph = torch.load(mmf_indirect(config.kg_path))
        self.graph, self.graph_idx, self.edge_index, self.edge_type = make_graph(
            raw_graph, config.prune_culdesacs
        )

        # Get all the useful graph attributes
        self.num_nodes = len(self.graph.nodes)
        assert len(self.graph_idx.nodes) == self.num_nodes
        self.num_edges = len(self.graph.edges)
        assert len(self.graph_idx.edges) == self.num_edges
        assert self.edge_index.shape[1] == self.num_edges
        assert self.edge_type.shape[0] == self.num_edges
        self.num_relations = len(raw_graph["relations2idx"])

        # Get the dataset specific info and relate it to the constructed graph
        (
            self.name2node_idx,
            self.qid2nodeact,
            self.img_class_sz,
        ) = self.get_dataset_info(config)

        # And get the answer related info
        (
            self.index_in_ans,
            self.index_in_node,
            self.graph_answers,
            self.graph_ans_node_idx,
        ) = self.get_answer_info(config)

        # Save graph answers (to be used by data loader)
        torch.save(self.graph_answers, mmf_indirect(config.graph_vocab_file))

        # If features have w2v, initialize it here
        node2vec_filename = mmf_indirect(config.node2vec_filename)
        node_names = list(self.name2node_idx.keys())
        valid_node2vec = False
        if os.path.exists(node2vec_filename):
            with open(node2vec_filename, "rb") as f:
                node2vec, node_names_saved, no_match_nodes = pickle.load(f)

            # Make sure the nodes here are identical (otherwise,
            # when we update graph code, we might have the wrong graph)
            if set(node_names) == set(node_names_saved):
                valid_node2vec = True

        # Generate node2vec if not done already
        if not valid_node2vec:
            node2vec, node_names_dbg, no_match_nodes = prepare_embeddings(
                node_names,
                mmf_indirect(config.embedding_file),
                config.add_w2v_multiword,
            )
            print("Saving synonym2vec to pickle file:", node2vec_filename)
            pickle.dump(
                (node2vec, node_names_dbg, no_match_nodes),
                open(node2vec_filename, "wb"),
            )

        # Get size
        self.w2v_sz = node2vec[list(node2vec.keys())[0]].shape[0]

        # Get node input dim
        self.in_node_dim = 0
        self.q_offest = 0
        self.img_offset = 0
        self.vb_offset = 0
        self.q_enc_offset = 0
        self.w2v_offset = 0

        # Add question (size 1)
        if "question" in config.node_inputs:
            self.q_offset = self.in_node_dim
            self.in_node_dim += 1

        # Add classifiers
        if "classifiers" in config.node_inputs:
            self.img_offset = self.in_node_dim
            self.in_node_dim += self.img_class_sz

        # Add w2v
        if "w2v" in config.node_inputs:
            self.w2v_offset = self.in_node_dim
            self.in_node_dim += self.w2v_sz

        # Doing no w2v as a seperate option to make this code a LOT simpler
        self.use_w2v = config.use_w2v
        if self.use_w2v:
            # Create the base node feature matrix
            # torch.Tensor of size num_nodes x in_node_dim
            # In forward pass, will need to copy this batch_size times and
            # convert to cuda
            self.base_node_features = torch.zeros(self.num_nodes, self.in_node_dim)

            # Copy over w2v
            for node_name in node2vec:
                # Get w2v, convert to torch, then copy over
                w2v = torch.from_numpy(node2vec[node_name])
                node_idx = self.name2node_idx[node_name]
                self.base_node_features[
                    node_idx, self.w2v_offset : self.w2v_offset + self.w2v_sz
                ].copy_(w2v)
        else:
            self.in_node_dim -= self.w2v_sz
            self.base_node_features = torch.zeros(self.num_nodes, self.in_node_dim)

        # Init
        full_node_dim = self.in_node_dim
        special_input_node = False
        special_input_sz = None

        # If feed_special_node, set inputs to graph network
        if (
            "feed_special_node" in self.config_extra
            and self.config_extra["feed_special_node"]
        ):
            assert not self.config_extra["compress_crossmodel"]
            special_input_node = True
            special_input_sz = 0

            # Get input size
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_logit_to_graph"
            ):
                special_input_sz += self.config.num_labels
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_hid_to_graph"
            ):
                special_input_sz += self.config_extra["vb_hid_sz"]
            if (
                "feed_q_to_graph" in self.config_extra
                and self.config_extra["feed_q_to_graph"]
            ):
                special_input_sz += self.config_extra["q_hid_sz"]

        # Otherwise, we feed into every graph node at start
        else:
            # Add vb conf (just the conf)
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_logit_to_graph"
            ):
                assert not self.config_extra["compress_crossmodel"]
                self.vb_offset = self.in_node_dim
                full_node_dim += 1

            # Add vb vector
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_hid_to_graph"
            ):
                self.vb_offset = self.in_node_dim
                if self.config_extra["compress_crossmodel"]:
                    full_node_dim += self.config_extra["crossmodel_compress_dim"]

                    # Make a compress layer (just a linear tranform)
                    self.compress_linear = nn.Linear(
                        self.config_extra["vb_hid_sz"],
                        self.config_extra["crossmodel_compress_dim"],
                    )

                else:
                    full_node_dim += self.config_extra["vb_hid_sz"]

            # Add q vector
            if (
                "feed_q_to_graph" in self.config_extra
                and self.config_extra["feed_q_to_graph"]
            ):
                assert not self.config_extra["compress_crossmodel"]
                self.q_enc_offset = self.in_node_dim
                full_node_dim += self.config_extra["q_hid_sz"]

        # Set noback_vb
        self.noback_vb = self.config_extra["noback_vb"]

        # Convert edge_index and edge_type matrices to torch
        # In forward pass, we repeat this by bs and convert to cuda
        self.edge_index = torch.from_numpy(self.edge_index)
        self.edge_type = torch.from_numpy(self.edge_type)

        # These are the forward pass data inputs to graph network
        # They are None to start until we know the batch size
        self.node_features_forward = None
        self.edge_index_forward = None
        self.edge_type_forward = None

        # Make graph network itself
        self.gn = GraphNetwork(
            config,
            full_node_dim,
            self.num_relations,
            self.num_nodes,
            special_input_node=special_input_node,
            special_input_sz=special_input_sz,
        )

        # Init hidden debug (used for analysis)
        self.graph_hidden_debug = None

    def get_dataset_info(self, config):
        # Load dataset info
        dataset_data = torch.load(mmf_indirect(config.dataset_info_path))

        # Go through and collect symbol names and confs from our pretrained classifiers
        # Hardcoded to the classifiers
        qid2qnode = {}
        qid2imginfo = {}
        for dat in dataset_data:
            # Get qid
            qid = dat["id"]

            # Get q symbols
            q_words = list(dat["symbols_q"])
            qid2qnode[qid] = q_words

            # Get confidences
            in_data = dat["in_names_confs"]
            in_data = [(name, conf, 0) for name, conf in in_data]
            places_data = dat["places_names_confs"]
            places_data = [(name, conf, 1) for name, conf in places_data]
            lvis_data = dat["lvis_names_confs"]
            lvis_data = [(name, conf, 2) for name, conf in lvis_data]
            vg_data = dat["vg_names_confs"]
            vg_data = [(name, conf, 3) for name, conf in vg_data]
            all_image_tuples = in_data + places_data + lvis_data + vg_data

            # Make into dict to start (name -> conf Tensor)
            img_data = {}
            for name, conf, datasetind in all_image_tuples:
                # Check if name has been put in yet
                if name in img_data:
                    # If yes, insert new confidence in the right place
                    # Don't overwrite in same ind unless conf is higher
                    if conf > img_data[name][datasetind].item():
                        img_data[name][datasetind] = conf
                else:
                    # Otherwise, all zeros and add conf to the right index
                    conf_data = torch.zeros(4)
                    conf_data[datasetind] = conf
                    img_data[name] = conf_data

            # Convert dict to tuples list and add to qid dict
            img_data = [(name, img_data[name]) for name in img_data]
            qid2imginfo[qid] = img_data

        # Convert qid2qnode and qid2imginfo to go from qid -> (name, conf)
        # to qid -> (node_idx, conf) and merge q and img info (concat)
        name2node_idx = {}
        idx = 0
        for nodename in self.graph.nodes:
            name2node_idx[nodename] = idx
            idx += 1
        qid2nodeact = {}
        img_class_sz = None
        for qid in qid2qnode:
            # Get words / confs
            q_words = qid2qnode[qid]  # qid -> [qw_1, qw_2, ...]
            # qid -> [(iw_1, conf_c1, conf_c2, ...), ...]
            img_info = qid2imginfo[qid]
            img_words = [x[0] for x in img_info]
            img_confs = [x[1] for x in img_info]

            # Get the node feature size
            if img_class_sz is None:
                # img_class_confs = img_confs[0]
                assert type(img_confs[0]) is torch.Tensor
                img_class_sz = img_confs[0].size(0)

            # We will arrange the node info
            # [q, img_class_1_conf, img_class_2_conf ... w2v]
            # Add to list
            node_info = {}  # node_idx -> torch.Tensor(q, ic1, ic2, ...)
            for word in q_words:
                # Continue if q word is not in the graph
                if word not in name2node_idx:
                    continue

                # Add node info
                node_idx = name2node_idx[word]
                val = torch.zeros(img_class_sz + 1)
                val[0] = 1
                node_info[node_idx] = val

            # Add img info to node info
            for word, img_confs_w in zip(img_words, img_confs):
                # Continue if img word not in graph
                if word not in name2node_idx:
                    continue

                node_idx = name2node_idx[word]
                if node_idx in node_info:
                    # Append class info to existing node info
                    node_info[node_idx][1:].copy_(img_confs_w)
                else:
                    # Just prepend a zero to the img info (not a question word)
                    val = torch.zeros(img_class_sz + 1)
                    val[1:].copy_(img_confs_w)
                    node_info[node_idx] = val

            # Add node info to dict
            # This structure will be used to dynamically create node info
            # during forward pass
            qid2nodeact[qid] = node_info

        # Check the average # of node activations is reasonable
        num_acts_per_qid = np.mean(
            [len(qid2nodeact[qid].keys()) for qid in qid2nodeact]
        )
        print("Average of %f nodes activated per question" % num_acts_per_qid)

        # Return
        return name2node_idx, qid2nodeact, img_class_sz

    # Get answer info
    def get_answer_info(self, config):
        # Get answer info
        # Recreates mmf answer_vocab here essentially
        answer_vocab = VocabDict(mmf_indirect(config.vocab_file))
        assert len(answer_vocab) == config.num_labels

        # If we're in okvqa v1.0, need to do this a bit differently
        if config.okvqa_v_mode in ["v1.0", "v1.0-121", "v1.0-121-mc"]:
            # Load the answer translation file (to go from raw strings to
            # stemmed in v1.0 vocab)
            tx_data = torch.load(mmf_indirect(config.ans_translation_file))
            if config.okvqa_v_mode in ["v1.0-121", "v1.0-121-mc"]:
                old_graph_vocab = torch.load(mmf_indirect(config.old_graph_vocab_file))

            # Get a list of answer node indices
            # Important if we want to index those out to (for instance)
            # do node classification on them
            index_in_ans = []
            index_in_node = []
            graph_answers = []
            nomatch = []
            for ans_str in answer_vocab.word2idx_dict:
                # Regular, don't worry about 1-1
                if config.okvqa_v_mode == "v1.0":
                    # Convert it to the most common raw answer and
                    # see if it's in the graph
                    if ans_str not in tx_data["v10_2_v11_mc"]:
                        nomatch.append(ans_str)
                        continue

                    # Try most common
                    if tx_data["v10_2_v11_mc"][ans_str] in self.name2node_idx:
                        # Get raw answer string
                        raw_ans = tx_data["v10_2_v11_mc"][ans_str]
                    else:
                        # Otherwise try all other options
                        v11_counts = tx_data["v10_2_v11_count"][ans_str]
                        sorted_counts = sorted(
                            v11_counts.items(), key=lambda x: x[1], reverse=True
                        )
                        raw_ans = None
                        for k, _ in sorted_counts:
                            if k in self.name2node_idx:
                                raw_ans = k
                                break

                        # If still no match, continue
                        if raw_ans is None:
                            nomatch.append(ans_str)
                            continue

                    # Add ans_str to graph answers
                    graph_answers.append(ans_str)

                    # Get the node index
                    # Use the raw name since that's what matches to nodes
                    node_idx = self.name2node_idx[raw_ans]
                    index_in_node.append(node_idx)

                    # Get the vocab index
                    ans_idx = answer_vocab.word2idx(ans_str)
                    index_in_ans.append(ans_idx)

                else:
                    # Convert it to the most common raw answer and see if
                    # it's in the graph
                    if ans_str not in tx_data["v10_2_v11_mc"]:
                        nomatch.append(ans_str)
                        continue

                    # Try raw too
                    if config.okvqa_v_mode == "v1.0-121-mc":
                        # Try most common
                        if tx_data["v10_2_raw_mc"][ans_str] in self.name2node_idx:
                            # Get raw answer string
                            raw_ans = tx_data["v10_2_raw_mc"][ans_str]
                        else:
                            # Otherwise try all other options
                            v11_counts = tx_data["v10_2_raw_count"][ans_str]
                            sorted_counts = sorted(
                                v11_counts.items(), key=lambda x: x[1], reverse=True
                            )
                            raw_ans = None
                            for k, _ in sorted_counts:
                                if k in self.name2node_idx:
                                    raw_ans = k
                                    break

                            # If still no match, continue
                            if raw_ans is None:
                                nomatch.append(ans_str)
                                continue
                    else:
                        # Try most common
                        if (
                            tx_data["v10_2_v11_mc"][ans_str] in self.name2node_idx
                            and tx_data["v10_2_v11_mc"][ans_str] in old_graph_vocab
                        ):
                            # Get raw answer string
                            raw_ans = tx_data["v10_2_v11_mc"][ans_str]
                        else:
                            # Otherwise try all other options
                            v11_counts = tx_data["v10_2_v11_count"][ans_str]
                            sorted_counts = sorted(
                                v11_counts.items(), key=lambda x: x[1], reverse=True
                            )
                            raw_ans = None
                            for k, _ in sorted_counts:
                                if k in self.name2node_idx and k in old_graph_vocab:
                                    raw_ans = k
                                    break

                            # If still no match, continue
                            if raw_ans is None:
                                nomatch.append(ans_str)
                                continue

                    # Check 1 to 1
                    if self.name2node_idx[raw_ans] in index_in_node:
                        if config.okvqa_v_mode == "v1.0-121-mc":
                            # Check which is more common
                            assert len(index_in_node) == len(graph_answers)
                            assert len(index_in_ans) == len(graph_answers)
                            idx = index_in_node.index(self.name2node_idx[raw_ans])
                            node_idx = index_in_node[idx]
                            old_ans_str = graph_answers[idx]
                            raw_counts = tx_data["v11_2_raw_count"][raw_ans]
                            assert ans_str in raw_counts and old_ans_str in raw_counts
                            assert ans_str != old_ans_str

                            # If new answer more common, go back and replace everything
                            if raw_counts[ans_str] > raw_counts[old_ans_str]:
                                assert node_idx == self.name2node_idx[raw_ans]
                                graph_answers[idx] = ans_str
                                ans_idx = answer_vocab.word2idx(ans_str)
                                index_in_ans[idx] = ans_idx
                            else:
                                continue
                        else:
                            nomatch.append(ans_str)
                            continue
                    else:
                        # Add ans_str to graph answers
                        graph_answers.append(ans_str)

                        # Get the node index
                        # Use the raw name since that's what matches to nodes
                        node_idx = self.name2node_idx[raw_ans]
                        index_in_node.append(node_idx)

                        # Get the vocab index
                        ans_idx = answer_vocab.word2idx(ans_str)
                        index_in_ans.append(ans_idx)
            print("%d answers not matches" % len(nomatch))

            # Get node indices for alphabetized graph answer too
            graph_answers = sorted(graph_answers)
            graph_ans_node_idx = []
            for ans_str in graph_answers:
                # Get node index
                node_idx = self.name2node_idx[raw_ans]
                graph_ans_node_idx.append(node_idx)
        else:
            assert config.okvqa_v_mode == "v1.1"

            # Get a list of answer node indices
            # Important if we want to index those out to (for instance)
            # do node classification on them
            index_in_ans = []
            index_in_node = []
            graph_answers = []
            for ans_str in answer_vocab.word2idx_dict:
                # Check if it's in the graph
                if ans_str not in self.name2node_idx:
                    continue

                # Add ans_str to graph answers
                graph_answers.append(ans_str)

                # Get the node index
                node_idx = self.name2node_idx[ans_str]
                index_in_node.append(node_idx)

                # Get the vocab index
                ans_idx = answer_vocab.word2idx(ans_str)
                index_in_ans.append(ans_idx)

            # Get node indices for alphabetized graph answer too
            graph_answers = sorted(graph_answers)
            graph_ans_node_idx = []
            for ans_str in graph_answers:
                # Get node index
                node_idx = self.name2node_idx[ans_str]
                graph_ans_node_idx.append(node_idx)

        # Sanity checks
        # Should be same length
        assert len(index_in_ans) == len(index_in_node)
        # And no repeats
        assert len(index_in_ans) == len(set(index_in_ans))
        if config.okvqa_v_mode != "v1.0":
            assert len(index_in_node) == len(set(index_in_node))
        assert len(graph_answers) == len(graph_ans_node_idx)

        # Check that the overlap is reasonable
        num_ans_in_graph = len(index_in_ans)
        print("%d answers in graph" % num_ans_in_graph)

        # Convert to tensors now
        index_in_ans = torch.LongTensor(index_in_ans)
        index_in_node = torch.LongTensor(index_in_node)
        graph_ans_node_idx = torch.LongTensor(graph_ans_node_idx)

        return index_in_ans, index_in_node, graph_answers, graph_ans_node_idx

    # Forward function
    # Converts from sample_list to the exact structure needed by the graph network
    # Assume right now that it's just passed in exactly how
    # I need it and I'll figure it out layer
    def forward(self, sample_list):
        # Get the batch size, qids, and device
        qids = sample_list["id"]
        batch_size = qids.size(0)
        device = qids.device

        # First, if this is first forward pass or batch size changed,
        # we need to allocate everything
        if (
            self.node_features_forward is None
            or batch_size * self.num_nodes != self.node_features_forward.size(0)
        ):
            # Allocate the data
            self.node_features_forward = torch.zeros(
                self.num_nodes * batch_size, self.in_node_dim
            ).to(device)
            _, num_edges = self.edge_index.size()
            self.edge_index_forward = (
                torch.LongTensor(2, num_edges * batch_size).fill_(0).to(device)
            )
            if self.gn.gcn_type == "RGCN":
                self.edge_type_forward = (
                    torch.LongTensor(num_edges * batch_size).fill_(0).to(device)
                )

            # Get initial values for data
            for batch_ind in range(batch_size):
                # Copy base_node_features without modification
                self.node_features_forward[
                    self.num_nodes * batch_ind : self.num_nodes * (batch_ind + 1), :
                ].copy_(self.base_node_features)

                # Copy edge_index, but we add self.num_nodes*batch_ind to every value
                # This is equivalent to batch_size independent subgraphs
                self.edge_index_forward[
                    :, batch_ind * num_edges : (batch_ind + 1) * num_edges
                ].copy_(self.edge_index)
                self.edge_index_forward[
                    :, batch_ind * num_edges : (batch_ind + 1) * num_edges
                ].add_(batch_ind * self.num_nodes)

                # And copy edge_types without modification
                if self.gn.gcn_type == "RGCN":
                    self.edge_type_forward[
                        batch_ind * num_edges : (batch_ind + 1) * num_edges
                    ].copy_(self.edge_type)

        # Zero fill the confidences for node features
        assert (
            self.w2v_offset is not None
            and self.q_offset is not None
            and self.img_offset is not None
        )
        assert self.w2v_offset > 0
        self.node_features_forward[:, : self.w2v_offset].zero_()

        # If in not using confs mode, just leave these values at zero
        if not self.config.use_conf:
            pass
        elif not self.config.use_q:
            assert self.config.use_img

            # Fill in the new confidences for this batch based on qid
            all_node_idx = []
            for batch_ind, qid in enumerate(qids):
                # Fill in the activated nodes into node_features
                # These always start at zero
                node_info = self.qid2nodeact[qid.item()]
                for node_idx in node_info:
                    node_val = node_info[node_idx]
                    # Zero-out q
                    node_val[0] = 0
                    self.node_features_forward[
                        self.num_nodes * batch_ind + node_idx,
                        : self.img_offset + self.img_class_sz,
                    ].copy_(node_val)
                    all_node_idx.append(node_idx)

        elif not self.config.use_img:
            # Fill in the new confidences for this batch based on qid
            all_node_idx = []
            for batch_ind, qid in enumerate(qids):
                # Fill in the activated nodes into node_features
                # These always start at zero
                node_info = self.qid2nodeact[qid.item()]
                for node_idx in node_info:
                    node_val = node_info[node_idx]

                    # Zero-out img
                    node_val[1] = 0
                    node_val[2] = 0
                    node_val[3] = 0
                    node_val[4] = 0
                    self.node_features_forward[
                        self.num_nodes * batch_ind + node_idx,
                        : self.img_offset + self.img_class_sz,
                    ].copy_(node_val)
                    all_node_idx.append(node_idx)
        elif self.config.use_partial_img:
            # Get the index of image we're keeping
            # For all confs except partial_img_idx, fill in 0's
            assert self.config.partial_img_idx in [0, 1, 2, 3]

            # Fill in the new confidences for this batch based on qid
            all_node_idx = []
            for batch_ind, qid in enumerate(qids):
                # Fill in the activated nodes into node_features
                # These always start at zero
                node_info = self.qid2nodeact[qid.item()]
                for node_idx in node_info:
                    node_val = node_info[node_idx]
                    # Zero-out img (except for one)
                    db_count = 0
                    if self.config.partial_img_idx != 0:
                        node_val[1] = 0
                        db_count += 1
                    if self.config.partial_img_idx != 1:
                        node_val[2] = 0
                        db_count += 1
                    if self.config.partial_img_idx != 2:
                        node_val[3] = 0
                        db_count += 1
                    if self.config.partial_img_idx != 3:
                        node_val[4] = 0
                        db_count += 1
                    assert db_count == 3
                    self.node_features_forward[
                        self.num_nodes * batch_ind + node_idx,
                        : self.img_offset + self.img_class_sz,
                    ].copy_(node_val)
                    all_node_idx.append(node_idx)
        else:
            # Fill in the new confidences for this batch based on qid
            all_node_idx = []
            for batch_ind, qid in enumerate(qids):
                # Fill in the activated nodes into node_features
                # These always start at zero
                node_info = self.qid2nodeact[qid.item()]
                for node_idx in node_info:
                    node_val = node_info[node_idx]
                    self.node_features_forward[
                        self.num_nodes * batch_ind + node_idx,
                        : self.img_offset + self.img_class_sz,
                    ].copy_(node_val)
                    all_node_idx.append(node_idx)

        # If necessary, pass in "output nodes" depending on output calculation
        # This for instance tells the gn which nodes to subsample
        if self.gn.output_type == "graph_level_ansonly":
            output_nodes = self.index_in_node  # These are node indices that are answers
        elif self.gn.output_type == "graph_level_inputonly":
            output_nodes = torch.LongTensor(
                all_node_idx
            )  # These are all non-zero nodes for the question
        else:
            output_nodes = None

        # If we're feeding in special node, need a different forward pass into self.gn
        if (
            "feed_special_node" in self.config_extra
            and self.config_extra["feed_special_node"]
        ):
            # Get special_node_input
            # Add vb conf (just the conf)
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_logit_to_graph"
            ):
                # Go through answer vocab and copy conf into it
                if self.noback_vb:
                    vb_logits = sample_list["vb_logits"].detach()
                else:
                    vb_logits = sample_list["vb_logits"]
                special_node_input = torch.sigmoid(vb_logits)

            # Add vb feats
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_hid_to_graph"
            ):
                if self.noback_vb:
                    special_node_input = sample_list["vb_hidden"].detach()
                else:
                    special_node_input = sample_list["vb_hidden"]

            # Add q enc feats
            if (
                "feed_q_to_graph" in self.config_extra
                and self.config_extra["feed_q_to_graph"]
            ):
                special_node_input = sample_list["q_encoded"]

            # Do actual graph forward pass
            if self.gn.gcn_type == "RGCN":
                output, spec_out = self.gn(
                    self.node_features_forward,
                    self.edge_index_forward,
                    self.edge_type_forward,
                    batch_size=batch_size,
                    output_nodes=output_nodes,
                    special_node_input=special_node_input,
                )
            elif self.gn.gcn_type in ["GCN", "SAGE"]:
                output, spec_out = self.gn(
                    self.node_features_forward,
                    self.edge_index_forward,
                    batch_size=batch_size,
                    output_nodes=output_nodes,
                    special_node_input=special_node_input,
                )

        # Otherwise, proceed normally
        else:
            # Build node_forward
            # Concat other stuff onto it
            node_feats_tmp = self.node_features_forward

            # Add other input types
            # Add vb conf (just the conf)
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_logit_to_graph"
            ):
                assert not self.config_extra["compress_crossmodel"]
                # Go through answer vocab and copy conf into it
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size, self.num_nodes, -1)
                )
                if self.noback_vb:
                    vb_logits = sample_list["vb_logits"].detach()
                else:
                    vb_logits = sample_list["vb_logits"]
                vb_confs = torch.sigmoid(vb_logits)
                vb_confs_graphindexed = torch.zeros(batch_size, self.num_nodes).to(
                    device
                )
                vb_confs_graphindexed[:, self.index_in_node] = vb_confs[
                    :, self.index_in_ans
                ]
                node_feats_tmp = torch.cat(
                    [node_feats_tmp, vb_confs_graphindexed.unsqueeze(2)], dim=2
                )
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size * self.num_nodes, -1)
                )

            # Add vb feats
            if (
                "feed_vb_to_graph" in self.config_extra
                and self.config_extra["feed_vb_to_graph"]
                and self.config_extra["feed_mode"] == "feed_vb_hid_to_graph"
            ):
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size, self.num_nodes, -1)
                )

                # Optionally compress vb_hidden
                if self.noback_vb:
                    vb_hid = sample_list["vb_hidden"].detach()
                else:
                    vb_hid = sample_list["vb_hidden"]
                if self.config_extra["compress_crossmodel"]:
                    vb_hid = F.relu(self.compress_linear(vb_hid))
                node_feats_tmp = torch.cat(
                    [
                        node_feats_tmp,
                        vb_hid.unsqueeze(1).repeat((1, self.num_nodes, 1)),
                    ],
                    dim=2,
                )
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size * self.num_nodes, -1)
                )

            # Add q enc feats
            if (
                "feed_q_to_graph" in self.config_extra
                and self.config_extra["feed_q_to_graph"]
            ):
                assert not self.config_extra["compress_crossmodel"]
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size, self.num_nodes, -1)
                )
                node_feats_tmp = torch.cat(
                    [
                        node_feats_tmp,
                        sample_list["q_encoded"]
                        .unsqueeze(1)
                        .repeat((1, self.num_nodes, 1)),
                    ],
                    dim=2,
                )
                node_feats_tmp = node_feats_tmp.reshape(
                    (batch_size * self.num_nodes, -1)
                )

            # Do actual graph forward pass
            if self.gn.gcn_type == "RGCN":
                output, spec_out = self.gn(
                    node_feats_tmp,
                    self.edge_index_forward,
                    self.edge_type_forward,
                    batch_size=batch_size,
                    output_nodes=output_nodes,
                )
            elif self.gn.gcn_type in ["GCN", "SAGE"]:
                output, spec_out = self.gn(
                    node_feats_tmp,
                    self.edge_index_forward,
                    batch_size=batch_size,
                    output_nodes=output_nodes,
                )

        # Do any reindexing we need
        if self.config.output_type == "hidden_ans":
            # Outputs graph hidden features, but re-indexes them to anser vocab
            # Same as graph_prediction, but before final prediction
            assert output.size(1) == self.num_nodes
            assert output.size(2) == self.config.node_hid_dim
            assert output.dim() == 3

            # If in graph_analysis mode, save the hidden states here
            if self.config_extra["analysis_mode"]:
                self.graph_hidden_debug = output

            # Reindex to match with self.graph_vocab
            if self.config.output_order == "alpha":
                output = output[:, self.graph_ans_node_idx, :]
                assert output.size(1) == len(self.graph_answers)
            else:
                assert self.config.output_order == "ans"

                # Re-index into answer_vocab
                outputs_tmp = torch.zeros(
                    batch_size, self.config.num_labels, self.config.node_hid_dim
                ).to(device)
                outputs_tmp[:, self.index_in_ans, :] = output[:, self.index_in_node, :]
                output = outputs_tmp

        elif self.config.output_type in [
            "graph_level",
            "graph_level_ansonly",
            "graph_level_inputonly",
        ]:
            pass
            # Do nothing here, fc will happen layer
        else:
            assert self.config.output_type == "graph_prediction"

            # Output is size of graph
            assert output.size(1) == self.num_nodes
            assert output.dim() == 2

            # Re-index
            if self.config.output_order == "alpha":
                output = output[:, self.graph_ans_node_idx]
                assert output.size(1) == len(self.graph_answers)
            else:
                assert self.config.output_order == "ans"

                # Re-index into answer_vocab
                logits = (
                    torch.zeros(batch_size, self.config.num_labels)
                    .fill_(-1e3)
                    .to(device)
                )
                logits[:, self.index_in_ans] = output[:, self.index_in_node]
                output = logits

        # If we generated a spec_out in graph network, put in sample
        # list for other modules to use
        if spec_out is not None:
            sample_list["graph_special_node_out"] = spec_out

        return output

    # Add stuff to output for various analysis
    def add_analysis_to_output(self, output):
        # Add graphicx graph so we can see what nodes were activated / see subgraphs
        output["graph"] = self.graph
        output["graph_idx"] = self.graph_idx

        # Add structs so we can easily convert between vocabs
        output["name2node_idx"] = self.name2node_idx
        output["node_acts"] = self.qid2nodeact
        output["index_in_ans"] = self.index_in_ans
        output["index_in_node"] = self.index_in_node
        output["graph_answers"] = self.graph_answers
        output["graph_ans_node_idx"] = self.graph_ans_node_idx

        output["graph_hidden_act"] = self.graph_hidden_debug.cpu()

        # Return output with new keys
        return output


# Graph network network
class GraphNetwork(nn.Module):
    def __init__(
        self,
        config,
        in_node_dim,
        num_relations,
        num_nodes,
        special_input_node=False,
        special_input_sz=None,
    ):
        super().__init__()
        # Get/set parameters
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        # Passed in from GraphNetworkModule which constructs the input features
        self.in_node_dim = in_node_dim
        self.node_hid_dim = config.node_hid_dim
        self.num_gcn_conv = config.num_gcn_conv
        self.use_bn = config.use_batch_norm
        self.use_drop = config.use_dropout
        self.output_type = config.output_type
        self.gcn_type = config.gcn_type
        if self.use_drop:
            self.drop_p = config.dropout_p
        if "output_dim" in config:
            self.output_dim = config.output_dim
        else:
            self.output_dim = self.node_hid_dim
        self.special_input_node = special_input_node
        self.special_input_sz = special_input_sz
        self.output_special_node = config.output_special_node

        # Make GCN and batchnorm layers
        if self.num_gcn_conv >= 1:
            # Try to add CompGCN at some point
            if self.gcn_type == "RGCN":
                self.conv1 = RGCNConv(
                    self.in_node_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv1 = GCNConv(self.in_node_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv1 = SAGEConv(self.in_node_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)
        if self.num_gcn_conv >= 2:
            if self.use_bn:
                self.bn1 = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv2 = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv2 = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv2 = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)
        if self.num_gcn_conv >= 3:
            if self.use_bn:
                self.bn2 = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv3 = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv3 = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv3 = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)
        if self.num_gcn_conv >= 4:
            if self.use_bn:
                self.bn3 = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv4 = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv4 = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv4 = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)
        if self.num_gcn_conv >= 5:
            if self.use_bn:
                self.bn4 = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv5 = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv5 = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv5 = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)
        if self.num_gcn_conv >= 6:
            if self.use_bn:
                self.bn5 = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv6 = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv6 = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv6 = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)

        if self.num_gcn_conv >= 7:
            raise Exception("Did not implement %d gcn layers yet" % self.num_gcn_conv)

        # Add special node for input/output collection
        if self.output_special_node or self.special_input_node:
            # For special in (not mutally exclusive to special out)
            # Make an linear encoder to fit into node hid size
            if self.special_input_node:
                self.spec_input_fc = nn.Linear(self.special_input_sz, self.node_hid_dim)

            # Make graph conv (and transfer layers)
            # Add one to num_rels since we have a special relation for this
            if self.use_bn:
                self.bn_spec = BatchNorm(self.node_hid_dim)
            if self.gcn_type == "RGCN":
                self.conv_spec = RGCNConv(
                    self.node_hid_dim,
                    self.node_hid_dim,
                    self.num_relations + 1,
                    num_bases=None,
                )
            elif self.gcn_type == "GCN":
                self.conv_spec = GCNConv(self.node_hid_dim, self.node_hid_dim)
            elif self.gcn_type == "SAGE":
                self.conv_spec = SAGEConv(self.node_hid_dim, self.node_hid_dim)
            else:
                raise Exception("GCN type %s not implemented" % self.gcn_type)

            # On first pass, populate this and convert to cuda
            # Connects all node indices to the special node via a "special" edge type
            self.edge_index_special = None
            self.edge_type_special = None
            self.special_bs = None

        # Set output network
        if self.output_type in ["hidden", "hidden_subindex", "hidden_ans"]:
            # Don't really need anything here, either passing all of G,
            # or G for particular indices
            pass
        elif self.output_type in [
            "graph_level",
            "graph_level_ansonly",
            "graph_level_inputonly",
        ]:
            # Will first predict a logit for each node, then do
            # softmax addition of all graph features
            self.logit_pred = nn.Linear(
                self.node_hid_dim, 1
            )  # Predicts hid_dim -> 1, which then gets passed into softmax
            self.feat_layer = nn.Linear(self.node_hid_dim, self.output_dim)
        elif self.output_type in ["graph_prediction"]:
            # Need just a final logits prediction
            self.logit_pred = nn.Linear(self.node_hid_dim, 1)
        else:
            raise Exception(
                "Output type %s is not implemented right now" % self.output_type
            )

    def forward(
        self,
        x,
        edge_index,
        edge_type=None,
        batch_size=1,
        output_nodes=None,
        special_node_input=None,
    ):
        # x is the input node features num_nodesxin_feat
        # edge_index is a 2xnum_edges matrix of which nodes each edge connects
        # edge_type is a num_edges of what the edge type is for each of those types
        if self.num_nodes is not None:
            assert x.size(0) == self.num_nodes * batch_size

        # Set optional spec_out to None
        spec_out = None

        # Check type and inputs match
        if self.gcn_type == "RGCN":
            assert edge_type is not None
        elif self.gcn_type in ["GCN", "SAGE"]:
            assert edge_type is None
        else:
            raise Exception("GCN type %s not implemented" % self.gcn_type)

        # First GCN conv
        if edge_type is not None:
            x = self.conv1(x, edge_index, edge_type)
        else:
            x = self.conv1(x, edge_index)
        if self.num_gcn_conv > 1:
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn1(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Second layer
            if edge_type is not None:
                x = self.conv2(x, edge_index, edge_type)
            else:
                x = self.conv2(x, edge_index)

        if self.num_gcn_conv > 2:
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn2(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Third layer
            if edge_type is not None:
                x = self.conv3(x, edge_index, edge_type)
            else:
                x = self.conv3(x, edge_index)

        if self.num_gcn_conv > 3:
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn3(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Third layer
            if edge_type is not None:
                x = self.conv4(x, edge_index, edge_type)
            else:
                x = self.conv4(x, edge_index)

        if self.num_gcn_conv > 4:
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn4(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Third layer
            if edge_type is not None:
                x = self.conv5(x, edge_index, edge_type)
            else:
                x = self.conv5(x, edge_index)

        if self.num_gcn_conv > 5:
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn5(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Third layer
            if edge_type is not None:
                x = self.conv6(x, edge_index, edge_type)
            else:
                x = self.conv6(x, edge_index)

        assert self.num_gcn_conv <= 6

        # Add special conv layer for special node input/output
        if self.output_special_node or self.special_input_node:
            # Encode special input
            if self.special_input_node:
                assert special_node_input is not None
                special_node_input = self.spec_input_fc(special_node_input)
            # Or zero-pad it
            else:
                special_node_input = torch.zeros(batch_size, self.node_hid_dim).to(
                    x.device
                )

            # Create special edge_index, edge_type matrices
            if self.edge_index_special is None or self.special_bs != batch_size:
                # Set special_bs
                # This makes sure the prebuild edge_index/type has right batch size
                self.special_bs = batch_size

                # Figure out the special node edges
                # Do bidirectional just to be safe
                spec_edges = []
                for batch_ind in range(batch_size):
                    spec_node_idx = self.num_nodes * batch_size + batch_ind
                    spec_edges += [
                        [node_idx, spec_node_idx]
                        for node_idx in range(
                            self.num_nodes * batch_ind, self.num_nodes * (batch_ind + 1)
                        )
                    ]
                    spec_edges += [
                        [spec_node_idx, node_idx]
                        for node_idx in range(
                            self.num_nodes * batch_ind, self.num_nodes * (batch_ind + 1)
                        )
                    ]
                assert len(spec_edges) == self.num_nodes * batch_size * 2
                self.edge_index_special = (
                    torch.LongTensor(spec_edges).transpose(0, 1).to(x.device)
                )

                # Make edge type (if necessary)
                if self.gcn_type == "RGCN":
                    self.edge_type_special = (
                        torch.LongTensor(len(spec_edges))
                        .fill_(self.num_relations)
                        .to(x.device)
                    )  # edge type is special n+1 edge type

            # Forward through final special conv
            # Transfer layers + bn/drop
            if self.use_bn:
                x = self.bn_spec(x)
            x = F.relu(x)
            if self.use_drop:
                x = F.dropout(x, p=self.drop_p, training=self.training)

            # Special conv layer
            edge_index_tmp = torch.cat([edge_index, self.edge_index_special], dim=1)
            x = torch.cat([x, special_node_input], dim=0)
            if edge_type is not None:
                edge_type_tmp = torch.cat([edge_type, self.edge_type_special], dim=0)
                x = self.conv_spec(x, edge_index_tmp, edge_type_tmp)
            else:
                x = self.conv_spec(x, edge_index_tmp)

            # Output
            if self.num_nodes is not None:
                assert x.size(0) == self.num_nodes * batch_size + batch_size
            # If it's output special, get the output as those special
            # node hidden states
            if self.output_special_node:
                # Should be just the last (batch_size) nodes
                spec_out = x[self.num_nodes * batch_size :]
                assert spec_out.size(0) == batch_size

            # Otherwise, we want to remove the last batch_size nodes
            # (since we don't use them)
            x = x[: self.num_nodes * batch_size]
            assert x.size(0) == self.num_nodes * batch_size
        # Reshape output to batch size now
        # For dynamic graph, we don't do the reshape. It's the class
        # above's job to reshape this properly
        if self.num_nodes is not None:
            x = x.reshape(batch_size, self.num_nodes, self.node_hid_dim)

        # Prepare final output
        if self.output_type in ["hidden", "hidden_ans", "hidden_subindex"]:
            # Don't really need anything here, either passing all of G
            # Subindexing happens a level up
            pass
        elif self.output_type in [
            "graph_level",
            "graph_level_ansonly",
            "graph_level_inputonly",
        ]:
            # Relu
            x = F.relu(x)

            # Check shape of x is num_nodes x hid_size
            assert x.shape[2] == self.node_hid_dim

            # For ansonly or inputonly, use input output_nodes (LongTensor) to reindex x
            if self.output_type in ["graph_level_ansonly", "graph_level_inputonly"]:
                assert output_nodes is not None
                x = x[:, output_nodes, :]

            bs, num_node, _ = x.shape
            x = x.reshape(bs * num_node, self.node_hid_dim)

            # Get feat
            feat = self.feat_layer(x)
            feat = feat.reshape(bs, num_node, self.output_dim)

            # Forward through linear to 1
            logit = self.logit_pred(x)
            logit = logit.reshape(bs, num_node)
            logit = F.softmax(logit)

            # Get weighted sum of x!
            x = torch.bmm(logit.unsqueeze(1), feat).squeeze()
        elif self.output_type in ["graph_prediction"]:
            # Need just a final logits prediction
            x = F.relu(x)
            x = self.logit_pred(x)
            x = x.squeeze()  # Remove final dim
        else:
            raise Exception("output type not known %s" % self.output_type)

        # Return output
        return x, spec_out
