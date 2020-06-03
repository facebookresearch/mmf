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
import styles from './styles.module.css';

const features = [
  {
    title: <>Less Boilerplate</>,
    imageUrl: 'img/boilerplate.svg',
    description: (
      <>
        MMF is designed from ground up to let you focus on what matters i.e.
        your model by providing boilerplate stuff such as distributed training,
        common datasets and state-of-the-art pretrained baselines
        out-of-the-box.
      </>
    ),
  },
  {
    title: <>Powered by PyTorch</>,
    imageUrl: 'img/pytorch_logo.svg',
    description: (
      <>
        MMF is built on top of PyTorch that brings all of the goodies and the
        power in your hands. Since MMF isn&apos;t strongly opinionated, you can
        still use all of your PyTorch knowledge here.
      </>
    ),
  },
  {
    title: <>Modular and Composable</>,
    imageUrl: 'img/puzzle_pieces.svg',
    description: (
      <>
        MMF is created to be easily extensible and composable. Through our
        modular design, user can use particular components they care about from
        MMF. Our configuration system allows MMF to easily adapt to any user
        needs.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={classnames('col col--4', styles.feature, 'text--center')}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
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
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={classnames('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <div className="hero__title">
            <img
              className={classnames(styles.heroImg)}
              src="img/logo.svg"
              alt="MMF Logo"
            />
          </div>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={classnames(
                'button button--primary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/hello')}>
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
                {features.map(({title, imageUrl, description}) => (
                  <Feature
                    key={title}
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
