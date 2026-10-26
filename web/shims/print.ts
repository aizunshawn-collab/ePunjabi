// Web replacement for 'react-native-print'. Opens the recreated document in
// a new tab and triggers the browser's own print dialog, since there is no
// native print spooler to hand a file to from a browser.
import {getBlob} from './virtual-fs';

type PrintOptions = {filePath?: string; html?: string};

function extractFileSrc(html: string): string | null {
  const match = html.match(/src="file:\/\/(.*?)"/);
  return match ? match[1] : null;
}

async function print(options: PrintOptions): Promise<void> {
  let path: string | null = null;
  let mimeType = 'application/pdf';
  if (options.filePath) {
    path = options.filePath;
  } else if (options.html) {
    path = extractFileSrc(options.html);
    mimeType = 'image/png';
  }
  if (!path) {
    throw new Error('Nothing to print');
  }
  const blob = getBlob(path);
  if (!blob) {
    throw new Error(`No in-memory file registered for "${path}"`);
  }
  const typedBlob = new Blob([blob], {type: mimeType});
  const objectUrl = URL.createObjectURL(typedBlob);
  const printWindow = window.open(objectUrl, '_blank');
  printWindow?.addEventListener('load', () => printWindow.print());
}

export default {print};
