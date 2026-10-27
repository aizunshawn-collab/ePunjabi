// Web replacement for 'react-native-tts', backed by the browser's
// SpeechSynthesis API. Mirrors the subset of the real library's API surface
// TranslationAppMobile.tsx uses: setDefaultLanguage/Pitch/Rate, speak,
// getInitStatus, voices, addEventListener/removeAllListeners for
// 'tts-start'/'tts-finish'/'tts-cancel'.
type TtsEvent = 'tts-start' | 'tts-finish' | 'tts-cancel';

class TtsWeb {
  private language = 'en-US';
  private pitch = 1;
  private rate = 1;
  private listeners: Record<string, Array<() => void>> = {};

  async setDefaultLanguage(lang: string): Promise<void> {
    this.language = lang;
  }

  async setDefaultPitch(pitch: number): Promise<void> {
    this.pitch = pitch;
  }

  async setDefaultRate(rate: number, _isAndroidCompat?: boolean): Promise<void> {
    this.rate = rate;
  }

  async getInitStatus(): Promise<void> {
    // Nothing to initialize on web - speechSynthesis is ready synchronously.
  }

  async voices(): Promise<SpeechSynthesisVoice[]> {
    return this.loadedVoices();
  }

  // getVoices() returns [] until the browser has asynchronously loaded its
  // voice list (fired via the 'voiceschanged' event) - calling speak()
  // before that happens can silently produce no audio at all (no error
  // event either) on Chrome/Edge, which is why TTS seemed to randomly not
  // "talk back" on the web build. Wait for the list at least once per page
  // load, with a timeout in case a browser never fires the event.
  private voicesReady: Promise<SpeechSynthesisVoice[]> | null = null;
  private loadedVoices(): Promise<SpeechSynthesisVoice[]> {
    if (!window.speechSynthesis) {
      return Promise.resolve([]);
    }
    const existing = window.speechSynthesis.getVoices();
    if (existing.length > 0) {
      return Promise.resolve(existing);
    }
    if (!this.voicesReady) {
      this.voicesReady = new Promise(resolve => {
        const finish = () => resolve(window.speechSynthesis.getVoices());
        window.speechSynthesis.addEventListener('voiceschanged', finish, {once: true});
        setTimeout(finish, 2000);
      });
    }
    return this.voicesReady;
  }

  addEventListener(event: TtsEvent, callback: () => void): void {
    (this.listeners[event] ??= []).push(callback);
  }

  removeAllListeners(event: TtsEvent): void {
    delete this.listeners[event];
  }

  private emit(event: TtsEvent): void {
    this.listeners[event]?.forEach(cb => cb());
  }

  async speak(text: string): Promise<void> {
    if (!window.speechSynthesis) {
      return;
    }
    const voices = await this.loadedVoices();
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = this.language;
      // Explicitly assigning a matching voice is more reliable across
      // browsers than relying on .lang alone to pick one - some browsers
      // (and their default voice) will otherwise silently drop the request.
      const match =
        voices.find(v => v.lang.toLowerCase() === this.language.toLowerCase()) ??
        voices.find(v => v.lang.toLowerCase().startsWith(this.language.slice(0, 2).toLowerCase()));
      if (match) {
        utterance.voice = match;
      }
      utterance.pitch = this.pitch;
      utterance.rate = this.rate;
      utterance.onstart = () => this.emit('tts-start');
      utterance.onend = () => {
        this.emit('tts-finish');
        resolve();
      };
      utterance.onerror = () => {
        this.emit('tts-cancel');
        resolve();
      };
      window.speechSynthesis.speak(utterance);
    });
  }
}

export default new TtsWeb();
