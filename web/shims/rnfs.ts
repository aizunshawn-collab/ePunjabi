// Web replacement for 'react-native-fs'. There is no real filesystem in a
// browser, so paths are just string keys into the virtual-fs registry (see
// virtual-fs.ts), written to by the audio-recorder/document-picker shims and
// read back here exactly like the real app reads an actual file on disk.
import {getBlob, putBlob, readBlobAsBase64, base64ToBlob} from './virtual-fs';

const CachesDirectoryPath = '/virtual-cache';

async function readFile(path: string, _encoding: 'base64'): Promise<string> {
  return readBlobAsBase64(path);
}

async function writeFile(path: string, base64Data: string, _encoding: 'base64'): Promise<void> {
  // Mime type doesn't matter here - callers only ever read this back out as
  // base64 again (see handlePickedDocument), never render it as an <img>/<a>
  // directly from this registry.
  putBlob(path, base64ToBlob(base64Data, 'application/octet-stream'));
}

export default {
  CachesDirectoryPath,
  readFile,
  writeFile,
};

export {getBlob};
