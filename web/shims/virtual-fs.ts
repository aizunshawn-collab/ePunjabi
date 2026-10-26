// In-memory stand-in for a real filesystem (there isn't one in a browser).
// The audio-recorder/RNFS/image/document-picker/share/print/camera-roll web
// shims all read and write through this single registry, keyed by the same
// fake "path" strings the real app already generates (e.g.
// RNFS.CachesDirectoryPath + '/voice_input.m4a') - this version is just a
// string key for the registry instead of an actual files on disk.
const registry = new Map<string, Blob>();

export function putBlob(path: string, blob: Blob): void {
  registry.set(path, blob);
}

export function getBlob(path: string): Blob | undefined {
  return registry.get(path);
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Strip the "data:<mime>;base64," prefix - callers want raw base64,
      // matching RNFS.readFile's return shape.
      resolve(result.split(',')[1] ?? '');
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function readBlobAsBase64(path: string): Promise<string> {
  const blob = registry.get(path);
  if (!blob) {
    throw new Error(`No in-memory file registered for "${path}"`);
  }
  return blobToBase64(blob);
}

export function base64ToBlob(base64: string, mimeType: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], {type: mimeType});
}

export function fileToBase64(file: File | Blob): Promise<string> {
  return blobToBase64(file);
}
