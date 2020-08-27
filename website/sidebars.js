/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

function FBInternal(elements) {
  return process.env.FB_INTERNAL ? elements : [];
}
function FBInternalWithOssFallback(elements, fallback) {
  return process.env.FB_INTERNAL ? elements : fallback;
}

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
      'notes/dataset_zoo',
      'notes/model_zoo',
      'notes/pretrained_models',
      'notes/projects',
    ],
    Tutorials: [
      'tutorials/dataset',
      'tutorials/concat_bert',
      'tutorials/checkpointing',
      'tutorials/processors',
      'tutorials/slurm',
    ],
    ...FBInternal({
      'FB Internal': ['fb/devserver', 'fb/fblearner'],
    }),
    Challenges: [
      'challenges/hateful_memes_challenge',
      'challenges/textvqa_challenge',
      'challenges/vqa_challenge',
    ],
    Projects: ['projects/butd', 'projects/m4c', 'projects/movie_mcan'],
  },
};
