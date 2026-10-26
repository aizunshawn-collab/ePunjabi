import React from 'react';
import ReactDOM from 'react-dom/client';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import TranslationAppMobile from '../src/TranslationAppMobile';

function Root(): JSX.Element {
  return (
    <SafeAreaProvider>
      <TranslationAppMobile />
    </SafeAreaProvider>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(<Root />);
