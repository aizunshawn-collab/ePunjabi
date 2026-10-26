// Web replacement for 'react-native-document-picker'. Uses a hidden
// <input type="file"> and stores the picked file's bytes in the virtual-fs
// registry (see virtual-fs.ts) under a generated "uri", so the existing
// RNFS.readFile(pickedFile.uri, 'base64') call in TranslationAppMobile.tsx
// works unchanged.
import {putBlob} from './virtual-fs';

class DocumentPickerCancelError extends Error {
  isDocumentPickerCancel = true;
}

type PickedFile = {uri: string; name: string; type: string};

let nextId = 0;

async function pick(options: {type?: string[]}): Promise<PickedFile[]> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    if (options.type?.includes('application/pdf')) {
      input.accept = 'application/pdf';
    }
    input.style.display = 'none';

    let settled = false;
    input.onchange = () => {
      const file = input.files?.[0];
      document.body.removeChild(input);
      settled = true;
      if (!file) {
        reject(new DocumentPickerCancelError('cancelled'));
        return;
      }
      const uri = `/virtual-picked-doc-${nextId++}`;
      putBlob(uri, file);
      resolve([{uri, name: file.name, type: file.type}]);
    };
    window.addEventListener(
      'focus',
      () =>
        setTimeout(() => {
          if (!settled) {
            settled = true;
            reject(new DocumentPickerCancelError('cancelled'));
          }
        }, 300),
      {once: true},
    );

    document.body.appendChild(input);
    input.click();
  });
}

function isCancel(error: unknown): boolean {
  return error instanceof DocumentPickerCancelError;
}

export default {
  pick,
  isCancel,
  types: {pdf: 'application/pdf'},
};
