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
    // Safari/WebKit (desktop and iOS) doesn't support 'audio/webm' at all -
    // MediaRecorder's constructor throws synchronously for an unsupported
    // mimeType, which previously made startRecorder() throw on every
    // Safari-based browser, silently falling back to the on-device
    // SpeechRecognition path (which can't auto-detect Punjabi vs English,
    // unlike the backend's real auto-detect - see startVoiceRecognition in
    // TranslationAppMobile.tsx). Try Safari's own supported type(s) too, and
    // as a last resort let the browser pick its own default rather than
    // throwing.
    const preferredTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/aac'];
    const mimeType = preferredTypes.find(type => MediaRecorder.isTypeSupported(type));
    this.mediaRecorder = mimeType ? new MediaRecorder(this.stream, {mimeType}) : new MediaRecorder(this.stream);
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
