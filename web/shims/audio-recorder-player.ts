// Web replacement for 'react-native-audio-recorder-player', backed by
// MediaRecorder. Recorded audio is kept in the virtual-fs registry (see
// virtual-fs.ts) under the same fake path the app already uses, so the
// RNFS.readFile() shim can hand it to the backend afterwards exactly like
// the real app does with an actual file on disk.
import {putBlob} from './virtual-fs';

export default class AudioRecorderPlayerWeb {
  private mediaRecorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];
  private currentPath = '';

  async startRecorder(path: string): Promise<string> {
    this.currentPath = path;
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
    this.mediaRecorder = new MediaRecorder(this.stream, {mimeType});
    this.mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };
    this.mediaRecorder.start();
    return path;
  }

  async stopRecorder(): Promise<string> {
    return new Promise(resolve => {
      if (!this.mediaRecorder) {
        resolve(this.currentPath);
        return;
      }
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, {type: this.mediaRecorder?.mimeType || 'audio/webm'});
        putBlob(this.currentPath, blob);
        this.stream?.getTracks().forEach(track => track.stop());
        resolve(this.currentPath);
      };
      this.mediaRecorder.stop();
    });
  }
}
