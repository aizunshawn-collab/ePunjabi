import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';
import { Alert, Platform } from 'react-native';

export type VoiceLanguage = 'en-US' | 'pa-IN';

export interface VoiceServiceCallbacks {
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;
  onSpeechResults?: (results: string[]) => void;
  onSpeechError?: (error: any) => void;
  onSpeechPartialResults?: (results: string[]) => void;
}

class VoiceService {
  private isListening: boolean = false;
  private callbacks: VoiceServiceCallbacks = {};

  constructor() {
    this.initializeVoice();
    this.initializeTts();
  }

  private initializeVoice() {
    Voice.onSpeechStart = this.onSpeechStart;
    Voice.onSpeechEnd = this.onSpeechEnd;
    Voice.onSpeechError = this.onSpeechError;
    Voice.onSpeechResults = this.onSpeechResults;
    Voice.onSpeechPartialResults = this.onSpeechPartialResults;
  }

  private initializeTts() {
    // Configure TTS settings
    Tts.setDefaultRate(0.5);
    Tts.setDefaultPitch(1.0);
    Tts.setDefaultLanguage('en-US');

    // TTS event listeners
    Tts.addEventListener('tts-start', () => {
      console.log('TTS started');
    });

    Tts.addEventListener('tts-finish', () => {
      console.log('TTS finished');
    });

    Tts.addEventListener('tts-cancel', () => {
      console.log('TTS cancelled');
    });
  }

  setCallbacks(callbacks: VoiceServiceCallbacks) {
    this.callbacks = callbacks;
  }

  private onSpeechStart = () => {
    console.log('Speech recognition started');
    this.isListening = true;
    this.callbacks.onSpeechStart?.();
  };

  private onSpeechEnd = () => {
    console.log('Speech recognition ended');
    this.isListening = false;
    this.callbacks.onSpeechEnd?.();
  };

  private onSpeechError = (error: any) => {
    console.log('Speech recognition error:', error);
    this.isListening = false;
    this.callbacks.onSpeechError?.(error);
  };

  private onSpeechResults = (event: any) => {
    console.log('Speech results:', event.value);
    this.callbacks.onSpeechResults?.(event.value);
  };

  private onSpeechPartialResults = (event: any) => {
    console.log('Speech partial results:', event.value);
    this.callbacks.onSpeechPartialResults?.(event.value);
  };

  async startListening(language: VoiceLanguage = 'en-US'): Promise<boolean> {
    try {
      if (this.isListening) {
        await this.stopListening();
      }

      // Check if speech recognition is available
      const isAvailable = await Voice.isAvailable();
      if (!isAvailable) {
        Alert.alert(
          'Speech Recognition Unavailable',
          'Speech recognition is not available on this device.'
        );
        return false;
      }

      await Voice.start(language);
      return true;
    } catch (error) {
      console.error('Error starting voice recognition:', error);
      Alert.alert(
        'Voice Recognition Error',
        'Failed to start voice recognition. Please try again.'
      );
      return false;
    }
  }

  async stopListening(): Promise<void> {
    try {
      await Voice.stop();
      this.isListening = false;
    } catch (error) {
      console.error('Error stopping voice recognition:', error);
    }
  }

  async cancelListening(): Promise<void> {
    try {
      await Voice.cancel();
      this.isListening = false;
    } catch (error) {
      console.error('Error cancelling voice recognition:', error);
    }
  }

  isCurrentlyListening(): boolean {
    return this.isListening;
  }

  async speak(text: string, language: VoiceLanguage = 'en-US'): Promise<void> {
    try {
      // Stop any ongoing speech
      await this.stopSpeaking();

      // Set language for TTS
      await Tts.setDefaultLanguage(language);

      // Speak the text
      await Tts.speak(text);
    } catch (error) {
      console.error('Error with text-to-speech:', error);
      Alert.alert(
        'Text-to-Speech Error',
        'Failed to speak the text. Please try again.'
      );
    }
  }

  async stopSpeaking(): Promise<void> {
    try {
      await Tts.stop();
    } catch (error) {
      console.error('Error stopping TTS:', error);
    }
  }

  async getAvailableLanguages(): Promise<string[]> {
    try {
      // Note: Voice recognition supported languages are platform-specific
      // Return common languages for now
      return ['en-US', 'pa-IN', 'en-GB', 'hi-IN'];
    } catch (error) {
      console.error('Error getting supported languages:', error);
      return [];
    }
  }

  async getTtsVoices(): Promise<any[]> {
    try {
      const voices = await Tts.voices();
      return voices;
    } catch (error) {
      console.error('Error getting TTS voices:', error);
      return [];
    }
  }

  destroy() {
    try {
      Voice.destroy();
      Tts.removeAllListeners('tts-start');
      Tts.removeAllListeners('tts-finish');
      Tts.removeAllListeners('tts-cancel');
    } catch (error) {
      console.error('Error destroying voice service:', error);
    }
  }
}

export default new VoiceService();