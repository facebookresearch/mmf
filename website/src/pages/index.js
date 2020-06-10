/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import classnames from 'classnames';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useThemeContext from '@theme/hooks/useThemeContext';
import styles from './styles.module.css';

const features = [
  {
    title: 'Less Boilerplate',
    imageUrl: 'img/boilerplate.svg',
    description: (
      <>
        MMF is designed from ground up to let you focus on what matters -- your
        model -- by providing boilerplate code for distributed training, common
        datasets and state-of-the-art pretrained baselines out-of-the-box.
      </>
    ),
  },
  {
    title: 'Powered by PyTorch',
    imageUrl: 'img/pytorch_logo.svg',
    description: (
      <>
        MMF is built on top of PyTorch that brings all of its power in your
        hands. MMF is not strongly opinionated. So you can use all of your
        PyTorch knowledge here.
      </>
    ),
  },
  {
    title: 'Modular and Composable',
    imageUrl: 'img/puzzle_pieces.svg',
    description: (
      <>
        MMF is created to be easily extensible and composable. Through our
        modular design, you can use specific components from MMF that you care
        about. Our configuration system allows MMF to easily adapt to your
        needs.
      </>
    ),
  },
];

function BannerImage() {
  const {isDarkTheme} = useThemeContext();
  const logoWhite = useBaseUrl('img/logo_white_text.svg');
  const logo = useBaseUrl('img/logo.svg');
  return (
    <img
      className={classnames(styles.heroImg)}
      src={isDarkTheme ? logoWhite : logo}
      alt="MMF Logo"
    />
  );
}

function Feature({imageUrl, title, description}) {
  const {isDarkTheme} = useThemeContext();
  const withoutExtension = imageUrl.split('.')[0];
  const whiteImageUrl = useBaseUrl(`${withoutExtension}_white.svg`);
  const normalImageUrl = useBaseUrl(imageUrl);
  const finalImageUrl = isDarkTheme ? whiteImageUrl : normalImageUrl;
  return (
    <div className={classnames('col col--4', styles.feature, 'text--center')}>
      {finalImageUrl && (
        <div className="text--center">
          <img
            className={styles.featureImage}
            src={finalImageUrl}
            alt={title}
          />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title=""
      description={
        'MMF is a modular framework powered by PyTorch for multimodal vision and ' +
        'language research from Facebook AI Research'
      }>
      <header className={classnames('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <div className="hero__title">
            <BannerImage />
          </div>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={classnames(
                'button button--primary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        {features && features.length && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map(({title, imageUrl, description}, idx) => (
                  <Feature
                    key={`feature${idx.toString()}`}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
