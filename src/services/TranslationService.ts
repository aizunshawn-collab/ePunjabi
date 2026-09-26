import AsyncStorage from '@react-native-async-storage/async-storage';
import API_CONFIG from '../config/apiConfig';

export interface TranslationResult {
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
}

class TranslationService {
  private static apiKey: string | null = API_CONFIG.GOOGLE_TRANSLATE_API_KEY;
  private static readonly API_URL = 'https://translation.googleapis.com/language/translate/v2';

  // Set Google Translate API key
  static async setApiKey(key: string): Promise<void> {
    this.apiKey = key;
    await AsyncStorage.setItem('google_translate_api_key', key);
  }

  // Get API key from storage
  static async getApiKey(): Promise<string | null> {
    if (this.apiKey) {
      return this.apiKey;
    }
    
    try {
      const storedKey = await AsyncStorage.getItem('google_translate_api_key');
      this.apiKey = storedKey;
      return storedKey;
    } catch (error) {
      console.error('Error getting API key from storage:', error);
      return null;
    }
  }

  // Translate text using Google Translate API
  static async translate(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
  ): Promise<string> {
    const apiKey = await this.getApiKey();
    
    if (!apiKey) {
      throw new Error('Google Translate API key not set. Please configure your API key.');
    }

    try {
      const response = await fetch(`${this.API_URL}?key=${apiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: text,
          source: sourceLanguage,
          target: targetLanguage,
          format: 'text',
        }),
      });

      if (!response.ok) {
        if (response.status === 400) {
          throw new Error('Invalid request. Please check your input text and language codes.');
        } else if (response.status === 403) {
          throw new Error('API key is invalid or quota exceeded. Please check your Google Cloud Console.');
        } else {
          throw new Error(`Translation service error: ${response.status}`);
        }
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(`Google Translate API error: ${data.error.message}`);
      }

      if (!data.data || !data.data.translations || data.data.translations.length === 0) {
        throw new Error('No translation result received');
      }

      return data.data.translations[0].translatedText;
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error. Please check your internet connection.');
    }
  }

  // Detect language of the input text
  static async detectLanguage(text: string): Promise<string> {
    const apiKey = await this.getApiKey();
    
    if (!apiKey) {
      throw new Error('Google Translate API key not set');
    }

    try {
      const response = await fetch(`https://translation.googleapis.com/language/translate/v2/detect?key=${apiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`Language detection failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(`Google Translate API error: ${data.error.message}`);
      }

      if (!data.data || !data.data.detections || data.data.detections.length === 0) {
        throw new Error('No language detection result received');
      }

      return data.data.detections[0][0].language;
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Language detection failed');
    }
  }

  // Get supported languages
  static async getSupportedLanguages(): Promise<any[]> {
    const apiKey = await this.getApiKey();
    
    if (!apiKey) {
      throw new Error('Google Translate API key not set');
    }

    try {
      const response = await fetch(`https://translation.googleapis.com/language/translate/v2/languages?key=${apiKey}&target=en`);
      
      if (!response.ok) {
        throw new Error(`Failed to get supported languages: ${response.status}`);
      }

      const data = await response.json();
      return data.data.languages;
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Failed to get supported languages');
    }
  }
}

export default TranslationService;