import React from 'react';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import TranslationAppMobile from './src/TranslationAppMobile';

const App = (): JSX.Element => {
  return (
    <SafeAreaProvider>
      <TranslationAppMobile />
    </SafeAreaProvider>
  );
};

export default App;