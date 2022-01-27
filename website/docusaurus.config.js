/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const {fbContent} = require('internaldocs-fb-helpers');

module.exports = {
  title: 'MMF',
  tagline:
    'A modular framework for vision & language multimodal ' +
    'research from Facebook AI Research (FAIR).',
  url: 'https://mmf.sh',
  baseUrl: '/',
  favicon: 'img/favicon.png',
  organizationName: 'facebookresearch',
  projectName: 'mmf',
  themeConfig: {
    image: 'img/logo.png',
    googleAnalytics: {
      trackingID: 'UA-135079836-3',
    },
    gtag: {
      trackingID: 'UA-135079836-3',
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
    navbar: {
      title: '',
      logo: {
        alt: 'MMF Logo',
        src: 'img/banner_logo.svg',
      },
      items: [
        {
          to: 'docs',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {
          to: 'api_redirect',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/facebookresearch/mmf',
          label: 'GitHub',
          position: 'left',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              href:
                'https://medium.com/pytorch/bootstrapping-a-multimodal-project-using-mmf-a-pytorch-powered-multimodal-framework-464f75164af7',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/facebookresearch/mmf',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
      ],
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'img/oss_logo.png',
        href: 'https://opensource.facebook.com',
      },
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          sidebarCollapsible: false,
          showLastUpdateAuthor: fbContent({
            internal: false,
            external: true,
          }),
          showLastUpdateTime: fbContent({
            internal: false,
            external: true,
          }),
          editUrl: fbContent({
            internal: 'https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/faim/mmf/website',
            external: 'https://github.com/facebookresearch/mmf/edit/main/website/'
          }),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        enableEditor: true,
      },
    ],
  ],
};
