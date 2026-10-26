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
    return window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
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
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = this.language;
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
