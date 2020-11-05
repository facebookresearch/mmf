/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const {fbInternalOnly} = require('internaldocs-fb-helpers');

module.exports = {
  docs: {
    'Getting started': [
      'getting_started/installation',
      'getting_started/features',
      'getting_started/quickstart',
      'getting_started/video_overview',
      'getting_started/faqs',
    ],
    Notes: [
      'notes/concepts',
      'notes/configuration',
      'notes/training_tricks',
      'notes/dataset_zoo',
      'notes/model_zoo',
      'notes/pretrained_models',
      'notes/projects',
    ],
    Tutorials: [
      'tutorials/dataset',
      'tutorials/concat_bert_tutorial',
      'tutorials/checkpointing',
      'tutorials/processors',
      'tutorials/slurm',
    ],
    ...fbInternalOnly({
      'FB Internal': ['fb/devserver', 'fb/fblearner'],
    }),
    Challenges: [
      'challenges/hateful_memes_challenge',
      'challenges/textvqa_challenge',
      'challenges/vqa_challenge',
    ],
    Projects: ['projects/butd', 'projects/m4c', 'projects/m4c_captioner', 'projects/movie_mcan'],
  },
};
