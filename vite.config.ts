import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Web build for the real TranslationAppMobile.tsx app via react-native-web.
// Every native-only package import is aliased to a browser-compatible shim
// under web/shims/ (see that folder for details on each one) - everything
// else (View, Text, StyleSheet, TouchableOpacity, Alert, Modal, etc.) is
// handled natively by react-native-web itself.
export default defineConfig({
  root: path.resolve(__dirname, 'web'),
  base: '/ePunjabi/app-web/',
  define: {
    __DEV__: JSON.stringify(false),
  },
  plugins: [
    react({
      babel: {
        babelrc: false,
        configFile: false,
      },
    }),
  ],
  resolve: {
    extensions: [
      '.web.tsx',
      '.web.ts',
      '.web.jsx',
      '.web.js',
      '.web.mjs',
      '.tsx',
      '.ts',
      '.jsx',
      '.js',
      '.mjs',
      '.json',
    ],
    alias: [
      {find: /^react-native$/, replacement: path.resolve(__dirname, 'web/shims/react-native.ts')},
      {find: '@react-native-voice/voice', replacement: path.resolve(__dirname, 'web/shims/voice.ts')},
      {find: 'react-native-tts', replacement: path.resolve(__dirname, 'web/shims/tts.ts')},
      {
        find: 'react-native-audio-recorder-player',
        replacement: path.resolve(__dirname, 'web/shims/audio-recorder-player.ts'),
      },
      {find: 'react-native-fs', replacement: path.resolve(__dirname, 'web/shims/rnfs.ts')},
      {
        find: 'react-native-image-picker',
        replacement: path.resolve(__dirname, 'web/shims/image-picker.ts'),
      },
      {
        find: 'react-native-document-picker',
        replacement: path.resolve(__dirname, 'web/shims/document-picker.ts'),
      },
      {find: 'react-native-share', replacement: path.resolve(__dirname, 'web/shims/share.ts')},
      {find: 'react-native-print', replacement: path.resolve(__dirname, 'web/shims/print.ts')},
      {
        find: '@react-native-camera-roll/camera-roll',
        replacement: path.resolve(__dirname, 'web/shims/camera-roll.ts'),
      },
      {
        find: '@react-native-async-storage/async-storage',
        replacement: path.resolve(__dirname, 'web/shims/async-storage.ts'),
      },
      {
        find: 'react-native-vector-icons/MaterialIcons',
        replacement: path.resolve(__dirname, 'web/shims/icon.tsx'),
      },
    ],
  },
  build: {
    outDir: path.resolve(__dirname, 'app-web'),
    emptyOutDir: true,
  },
});
