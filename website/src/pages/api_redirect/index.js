/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import {useHistory} from 'react-router-dom';

const API = () => {
  const history = useHistory();
  history.push('/');
  return (
    <BrowserOnly fallback={<p>Some Fallback Content</p>}>
      {() => {
        window.location.href = '/api';
      }}
    </BrowserOnly>
  );
};

export default API;
