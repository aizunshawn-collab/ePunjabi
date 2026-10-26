import React from 'react';
import {AppRegistry} from 'react-native';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import TranslationAppMobile from '../src/TranslationAppMobile';

function Root(): JSX.Element {
  return (
    <SafeAreaProvider>
      <TranslationAppMobile />
    </SafeAreaProvider>
  );
}

// AppRegistry.runApplication (rather than plain ReactDOM.createRoot) is what
// makes react-native-web stretch the root View to fill the viewport.
AppRegistry.registerComponent('App', () => Root);
AppRegistry.runApplication('App', {rootTag: document.getElementById('root')!});
