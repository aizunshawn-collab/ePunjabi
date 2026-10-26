// Web replacement for 'react-native', re-exporting react-native-web (the
// resolve.alias in vite.config.ts points plain 'react-native' imports here)
// plus a no-op PermissionsAndroid - the real module is Android-only and
// TranslationAppMobile.tsx only ever dereferences it inside
// `if (Platform.OS === 'android')` branches, which never run on web, but the
// import itself still needs something to bind to.
export * from 'react-native-web';

export const PermissionsAndroid = {
  PERMISSIONS: {
    RECORD_AUDIO: 'android.permission.RECORD_AUDIO',
    CAMERA: 'android.permission.CAMERA',
    READ_MEDIA_IMAGES: 'android.permission.READ_MEDIA_IMAGES',
    READ_EXTERNAL_STORAGE: 'android.permission.READ_EXTERNAL_STORAGE',
  },
  RESULTS: {
    GRANTED: 'granted',
    DENIED: 'denied',
    NEVER_ASK_AGAIN: 'never_ask_again',
  },
  request: async () => 'granted',
};
