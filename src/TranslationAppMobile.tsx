import React, {useState, useEffect} from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  PermissionsAndroid,
  Platform,
  TextInput,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import Voice from '@react-native-voice/voice';
import { GOOGLE_TRANSLATE_API_KEY } from '@env';

const API_KEY = GOOGLE_TRANSLATE_API_KEY;

const TranslationAppMobile = (): JSX.Element => {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState('Ready for voice translation');
  const [selectedVoice, setSelectedVoice] = useState('pa-IN-Wavenet-A');

  useEffect(() => {
    requestMicrophonePermission();
    initializeVoice();
    
    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  const requestMicrophonePermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Microphone Permission',
            message: 'This app needs access to your microphone for voice translation',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Permission Denied', 'Microphone permission is required for voice translation');
        }
      } catch (err) {
        console.warn(err);
      }
    }
  };

  const initializeVoice = () => {
    Voice.onSpeechStart = () => {
      setStatus('🎤 Listening...');
    };

    Voice.onSpeechEnd = () => {
      setStatus('Processing...');
    };

    Voice.onSpeechResults = (e) => {
      if (e.value && e.value[0]) {
        const spokenText = e.value[0];
        setInputText(spokenText);
        translateAndSpeak(spokenText);
      }
    };

    Voice.onSpeechError = (e) => {
      setStatus(`❌ Error: ${e.error?.message || 'Speech recognition failed'}`);
      setIsListening(false);
    };
  };

  const startVoiceRecognition = async () => {
    try {
      setIsListening(true);
      setInputText('');
      setTranslatedText('');
      setStatus('🎤 Tap to speak English...');
      await Voice.start('en-US');
    } catch (error) {
      console.error('Voice start error:', error);
      setStatus('❌ Failed to start voice recognition');
      setIsListening(false);
    }
  };

  const stopVoiceRecognition = async () => {
    try {
      await Voice.stop();
      setIsListening(false);
      setStatus('Processing your speech...');
    } catch (error) {
      console.error('Voice stop error:', error);
      setIsListening(false);
    }
  };

  const translateAndSpeak = async (text: string) => {
    setIsLoading(true);
    setStatus('🌐 Translating to Punjabi...');

    try {
      // Translate using Google Translate API
      const translateResponse = await fetch(
        `https://translation.googleapis.com/language/translate/v2?key=${API_KEY}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            q: text,
            source: 'en',
            target: 'pa',
            format: 'text',
          }),
        },
      );

      if (!translateResponse.ok) {
        throw new Error(`Translation failed: ${translateResponse.status}`);
      }

      const translateData = await translateResponse.json();
      const punjabi = translateData.data.translations[0].translatedText;
      setTranslatedText(punjabi);
      setStatus('🎵 Speaking Punjabi...');

      // Speak using Google Cloud TTS
      await speakPunjabi(punjabi);

    } catch (error) {
      console.error('Translation error:', error);
      setStatus(`❌ Translation failed: ${error}`);
      Alert.alert('Translation Error', 'Failed to translate. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const speakPunjabi = async (text: string) => {
    try {
      const ttsResponse = await fetch(
        `https://texttospeech.googleapis.com/v1/text:synthesize?key=${API_KEY}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            input: {text: text},
            voice: {
              languageCode: 'pa-IN',
              name: selectedVoice,
            },
            audioConfig: {
              audioEncoding: 'MP3',
              speakingRate: 0.9,
              pitch: 0,
            },
          }),
        },
      );

      if (!ttsResponse.ok) {
        throw new Error(`TTS failed: ${ttsResponse.status}`);
      }

      const ttsData = await ttsResponse.json();
      
      if (ttsData.audioContent) {
        // Note: In React Native, you'd need a library like react-native-sound
        // to play the base64 audio. For now, we'll just show success.
        setStatus('✅ Punjabi translation complete!');
        Alert.alert('Success', 'Translation complete! (Audio playback requires react-native-sound library)');
      }

    } catch (error) {
      console.error('TTS error:', error);
      setStatus(`❌ TTS failed: ${error}`);
    }
  };

  const testPhrases = [
    {punjabi: 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', english: 'Hello'},
    {punjabi: 'ਤੁਸੀ ਕਿਵੇਂ ਹੋ?', english: 'How are you?'},
    {punjabi: 'ਧੰਨਵਾਦ ਜੀ', english: 'Thank you'},
    {punjabi: 'ਮੈਂ ਠੀਕ ਹਾਂ', english: 'I am fine'},
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>🎵 Punjabi Voice Translator</Text>
          <Text style={styles.subtitle}>Mobile App</Text>
        </View>

        <View style={styles.voiceSection}>
          <TouchableOpacity
            style={[styles.voiceButton, isListening && styles.voiceButtonActive]}
            onPress={isListening ? stopVoiceRecognition : startVoiceRecognition}
            disabled={isLoading}>
            <Text style={styles.voiceButtonText}>
              {isListening ? '🛑 Stop Recording' : '🎤 Start Voice Translation'}
            </Text>
          </TouchableOpacity>
          
          <Text style={styles.voiceHint}>
            {isListening ? 'Speak now in English...' : 'Tap to speak English → Hear Punjabi'}
          </Text>
        </View>

        <View style={styles.statusCard}>
          <Text style={styles.statusText}>{status}</Text>
          {isLoading && <ActivityIndicator size="small" color="#4285f4" style={styles.loader} />}
        </View>

        <View style={styles.translationSection}>
          <View style={styles.textBox}>
            <Text style={styles.textBoxTitle}>🗣️ English</Text>
            <Text style={styles.textContent}>
              {inputText || 'Your English speech will appear here...'}
            </Text>
          </View>

          <View style={styles.textBox}>
            <Text style={styles.textBoxTitle}>🎯 Punjabi Translation</Text>
            <Text style={[styles.textContent, styles.punjabiText]}>
              {translatedText || 'ਤੁਹਾਡਾ ਪੰਜਾਬੀ ਅਨੁਵਾਦ ਇੱਥੇ ਦਿਖਾਈ ਦੇਵੇਗਾ...'}
            </Text>
          </View>
        </View>

        <View style={styles.voiceSelector}>
          <Text style={styles.sectionTitle}>🎵 Voice Selection</Text>
          <TouchableOpacity
            style={[styles.voiceOption, selectedVoice === 'pa-IN-Wavenet-A' && styles.voiceOptionSelected]}
            onPress={() => setSelectedVoice('pa-IN-Wavenet-A')}>
            <Text style={styles.voiceOptionText}>👩 Female (Premium)</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.voiceOption, selectedVoice === 'pa-IN-Wavenet-B' && styles.voiceOptionSelected]}
            onPress={() => setSelectedVoice('pa-IN-Wavenet-B')}>
            <Text style={styles.voiceOptionText}>👨 Male (Premium)</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.testSection}>
          <Text style={styles.sectionTitle}>🧪 Test Punjabi Voice</Text>
          <View style={styles.phrasesGrid}>
            {testPhrases.map((phrase, index) => (
              <TouchableOpacity
                key={index}
                style={styles.phraseButton}
                onPress={() => speakPunjabi(phrase.punjabi)}>
                <Text style={styles.phraseText}>{phrase.punjabi}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.features}>
          <Text style={styles.sectionTitle}>✨ App Features</Text>
          <Text style={styles.featureItem}>🎤 Voice recognition in English</Text>
          <Text style={styles.featureItem}>🌐 Real-time translation to Punjabi</Text>
          <Text style={styles.featureItem}>🗣️ Authentic Punjabi pronunciation</Text>
          <Text style={styles.featureItem}>📱 Native Android app experience</Text>
          <Text style={styles.featureItem}>🔒 Secure Google Cloud integration</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
    paddingVertical: 20,
    backgroundColor: 'white',
    borderRadius: 15,
    elevation: 2,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4285f4',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
  },
  voiceSection: {
    alignItems: 'center',
    marginVertical: 20,
  },
  voiceButton: {
    backgroundColor: '#34a853',
    paddingVertical: 20,
    paddingHorizontal: 30,
    borderRadius: 50,
    elevation: 5,
    minWidth: 250,
    alignItems: 'center',
  },
  voiceButtonActive: {
    backgroundColor: '#ea4335',
  },
  voiceButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  voiceHint: {
    marginTop: 10,
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  statusCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
    elevation: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statusText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  loader: {
    marginLeft: 10,
  },
  translationSection: {
    marginVertical: 15,
  },
  textBox: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginVertical: 8,
    elevation: 2,
    borderWidth: 2,
    borderColor: '#e9ecef',
  },
  textBoxTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#4285f4',
    marginBottom: 8,
  },
  textContent: {
    fontSize: 16,
    lineHeight: 24,
    color: '#333',
  },
  punjabiText: {
    fontSize: 18,
  },
  voiceSelector: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginVertical: 10,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  voiceOption: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginVertical: 5,
    borderWidth: 2,
    borderColor: '#e9ecef',
  },
  voiceOptionSelected: {
    borderColor: '#4285f4',
    backgroundColor: '#e8f0fe',
  },
  voiceOptionText: {
    fontSize: 14,
    color: '#333',
  },
  testSection: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginVertical: 10,
    elevation: 2,
  },
  phrasesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  phraseButton: {
    backgroundColor: '#e1f5fe',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
    marginVertical: 4,
  },
  phraseText: {
    fontSize: 16,
    color: '#01579b',
  },
  features: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginVertical: 10,
    elevation: 2,
  },
  featureItem: {
    fontSize: 14,
    color: '#333',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
});

export default TranslationAppMobile;
