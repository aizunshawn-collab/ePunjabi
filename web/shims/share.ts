// Web replacement for 'react-native-share'. Uses the Web Share API (for
// mobile browsers that support sharing files) falling back to a direct
// browser download, since most desktop browsers don't support sharing files.
import {getBlob} from './virtual-fs';

type ShareOptions = {url: string; type?: string; failOnCancel?: boolean};

function downloadBlob(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(objectUrl);
}

async function open({url, type}: ShareOptions): Promise<void> {
  const path = url.replace(/^file:\/\//, '');
  const blob = getBlob(path);
  if (!blob) {
    throw new Error(`No in-memory file registered for "${path}"`);
  }
  const mimeType = type || blob.type || 'application/octet-stream';
  const filename = path.split('/').pop() || 'document';
  const typedBlob = new Blob([blob], {type: mimeType});
  const file = new File([typedBlob], filename, {type: mimeType});

  if (navigator.canShare?.({files: [file]})) {
    await navigator.share({files: [file]});
    return;
  }
  downloadBlob(typedBlob, filename);
}

export default {open};
