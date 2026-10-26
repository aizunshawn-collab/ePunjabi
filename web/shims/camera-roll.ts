// Web replacement for '@react-native-camera-roll/camera-roll'. There is no
// photo gallery to save into from a browser, so this triggers a regular
// file download instead - only ever called for the recreated-document PNG
// (see saveDocumentResult in TranslationAppMobile.tsx, which routes PDFs
// through the share shim instead).
import {getBlob} from './virtual-fs';

async function save(uri: string, _options?: {type?: string}): Promise<string> {
  const path = uri.replace(/^file:\/\//, '');
  const blob = getBlob(path);
  if (!blob) {
    throw new Error(`No in-memory file registered for "${path}"`);
  }
  const typedBlob = new Blob([blob], {type: 'image/png'});
  const objectUrl = URL.createObjectURL(typedBlob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = 'translated_document.png';
  link.click();
  URL.revokeObjectURL(objectUrl);
  return uri;
}

export const CameraRoll = {save};
