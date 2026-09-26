import React, {useState, useEffect} from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import TranslationService from './services/TranslationService';
import VoiceService, { VoiceLanguage } from './services/VoiceService';
import ApiConfig from './components/ApiConfig';
import API_CONFIG from './config/apiConfig';

const TranslationApp = (): JSX.Element => {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState<'en' | 'pa'>('en');
  const [isLoading, setIsLoading] = useState(false);
  const [isApiKeySet, setIsApiKeySet] = useState(false);
  const [checkingApiKey, setCheckingApiKey] = useState(true);
  
  // Voice-related state
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');

  useEffect(() => {
    checkApiKey();
    initializeVoice();
    
    return () => {
      VoiceService.destroy();
    };
  }, []);

  const checkApiKey = async () => {
    try {
      // First try to use the configured API key
      if (API_CONFIG.GOOGLE_TRANSLATE_API_KEY) {
        await TranslationService.setApiKey(API_CONFIG.GOOGLE_TRANSLATE_API_KEY);
        setIsApiKeySet(true);
        return;
      }
      
      // Otherwise check if there's a stored API key
      const apiKey = await TranslationService.getApiKey();
      setIsApiKeySet(!!apiKey);
    } catch (error) {
      console.error('Error checking API key:', error);
    } finally {
      setCheckingApiKey(false);
    }
  };

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      Alert.alert('Error', 'Please enter text to translate');
      return;
    }

    setIsLoading(true);
    try {
      const targetLanguage = sourceLanguage === 'en' ? 'pa' : 'en';
      const result = await TranslationService.translate(
        inputText,
        sourceLanguage,
        targetLanguage,
      );
      setTranslatedText(result);
    } catch (error) {
      Alert.alert('Error', 'Translation failed. Please check your API key and try again.');
      console.error('Translation error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const swapLanguages = () => {
    setSourceLanguage(sourceLanguage === 'en' ? 'pa' : 'en');
    setInputText(translatedText);
    setTranslatedText(inputText);
  };

  const clearText = () => {
    setInputText('');
    setTranslatedText('');
  };

  // Voice functions
  const initializeVoice = async () => {
    // Request microphone permission on Android
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Microphone Permission',
            message: 'This app needs access to microphone for voice input',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert(
            'Permission Required',
            'Microphone permission is required for voice input functionality.'
          );
        }
      } catch (err) {
        console.warn('Permission request error:', err);
      }
    }

    // Set voice service callbacks
    VoiceService.setCallbacks({
      onSpeechStart: () => {
        setIsListening(true);
        setVoiceStatus('Listening... Speak now!');
      },
      onSpeechEnd: () => {
        setIsListening(false);
        setVoiceStatus('Processing speech...');
      },
      onSpeechResults: (results) => {
        if (results && results.length > 0) {
          setInputText(results[0]);
          setVoiceStatus('Speech captured successfully!');
          setTimeout(() => setVoiceStatus(''), 2000);
          // Auto-translate after capturing speech
          setTimeout(() => handleTranslate(), 500);
        }
      },
      onSpeechError: (error) => {
        setIsListening(false);
        setVoiceStatus('Speech recognition error');
        setTimeout(() => setVoiceStatus(''), 3000);
        console.error('Voice error:', error);
      },
      onSpeechPartialResults: (results) => {
        if (results && results.length > 0) {
          setVoiceStatus(`Hearing: "${results[0]}"`);
        }
      },
    });
  };

  const startVoiceInput = async () => {
    if (isListening) {
      await VoiceService.stopListening();
      return;
    }

    const voiceLanguage: VoiceLanguage = sourceLanguage === 'en' ? 'en-US' : 'pa-IN';
    const success = await VoiceService.startListening(voiceLanguage);
    
    if (!success) {
      setVoiceStatus('Voice recognition failed to start');
      setTimeout(() => setVoiceStatus(''), 3000);
    }
  };

  const speakInputText = async () => {
    if (!inputText.trim()) {
      Alert.alert('No Text', 'Please enter some text to speak.');
      return;
    }

    const voiceLanguage: VoiceLanguage = sourceLanguage === 'en' ? 'en-US' : 'pa-IN';
    setIsSpeaking(true);
    setVoiceStatus('Speaking input text...');
    
    try {
      await VoiceService.speak(inputText, voiceLanguage);
      setVoiceStatus('Finished speaking');
      setTimeout(() => setVoiceStatus(''), 2000);
    } catch (error) {
      setVoiceStatus('Speech playback failed');
      setTimeout(() => setVoiceStatus(''), 3000);
    } finally {
      setIsSpeaking(false);
    }
  };

  const speakTranslatedText = async () => {
    if (!translatedText.trim()) {
      Alert.alert('No Translation', 'Please translate some text first.');
      return;
    }

    const voiceLanguage: VoiceLanguage = sourceLanguage === 'en' ? 'pa-IN' : 'en-US';
    setIsSpeaking(true);
    setVoiceStatus('Speaking translation...');
    
    try {
      await VoiceService.speak(translatedText, voiceLanguage);
      setVoiceStatus('Finished speaking');
      setTimeout(() => setVoiceStatus(''), 2000);
    } catch (error) {
      setVoiceStatus('Speech playback failed');
      setTimeout(() => setVoiceStatus(''), 3000);
    } finally {
      setIsSpeaking(false);
    }
  };

  if (checkingApiKey) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!isApiKeySet) {
    return <ApiConfig onApiKeySet={() => setIsApiKeySet(true)} />;
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>English Translator</Text>
      
      <View style={styles.languageContainer}>
        <Text style={styles.languageLabel}>
          {sourceLanguage === 'en' ? 'English' : 'ਪੰਜਾਬੀ'}
        </Text>
        <TouchableOpacity style={styles.swapButton} onPress={swapLanguages}>
          <Text style={styles.swapButtonText}>⇄</Text>
        </TouchableOpacity>
        <Text style={styles.languageLabel}>
          {sourceLanguage === 'en' ? 'ਪੰਜਾਬੀ' : 'English'}
        </Text>
      </View>

      <ScrollView style={styles.scrollContainer}>
        <View style={styles.inputContainer}>
          <View style={styles.inputHeader}>
            <Text style={styles.inputLabel}>
              {sourceLanguage === 'en' ? 'English' : 'ਪੰਜਾਬੀ'}
            </Text>
            <View style={styles.voiceControls}>
              <TouchableOpacity
                style={[styles.voiceButton, isListening && styles.voiceButtonActive]}
                onPress={startVoiceInput}
                disabled={isLoading}>
                <Text style={styles.voiceButtonText}>🎤</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.voiceButton, isSpeaking && styles.voiceButtonActive]}
                onPress={speakInputText}
                disabled={isLoading || isSpeaking}>
                <Text style={styles.voiceButtonText}>🔊</Text>
              </TouchableOpacity>
            </View>
          </View>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder={
              sourceLanguage === 'en'
                ? 'Enter English text or tap microphone to speak...'
                : 'ਪੰਜਾਬੀ ਟੈਕਸਟ ਦਾਖਲ ਕਰੋ ਜਾਂ ਮਾਈਕ੍ਰੋਫ਼ੋਨ ਦਬਾਓ...'
            }
            multiline
            textAlignVertical="top"
          />
        </View>

        {voiceStatus ? (
          <View style={styles.voiceStatusContainer}>
            <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
          </View>
        ) : null}

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.translateButton]}
            onPress={handleTranslate}
            disabled={isLoading}>
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Translate</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.clearButton]}
            onPress={clearText}>
            <Text style={styles.buttonText}>Clear</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.outputContainer}>
          <View style={styles.inputHeader}>
            <Text style={styles.outputLabel}>Translation:</Text>
            <TouchableOpacity
              style={[styles.voiceButton, isSpeaking && styles.voiceButtonActive]}
              onPress={speakTranslatedText}
              disabled={isLoading || isSpeaking || !translatedText}>
              <Text style={styles.voiceButtonText}>🔊</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.translationBox}>
            <Text style={styles.translatedText}>{translatedText}</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  languageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
    paddingHorizontal: 20,
  },
  languageLabel: {
    fontSize: 18,
    fontWeight: '600',
    color: '#007AFF',
  },
  swapButton: {
    backgroundColor: '#007AFF',
    borderRadius: 25,
    width: 50,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  swapButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  scrollContainer: {
    flex: 1,
  },
  inputContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  textInput: {
    height: 120,
    fontSize: 16,
    textAlignVertical: 'top',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
  },
  button: {
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 25,
    minWidth: 100,
    alignItems: 'center',
  },
  translateButton: {
    backgroundColor: '#007AFF',
  },
  clearButton: {
    backgroundColor: '#FF6B6B',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  outputContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  outputLabel: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  translationBox: {
    minHeight: 100,
    padding: 12,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  translatedText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  inputHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  voiceControls: {
    flexDirection: 'row',
    gap: 10,
  },
  voiceButton: {
    backgroundColor: '#28a745',
    width: 45,
    height: 45,
    borderRadius: 22.5,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  voiceButtonActive: {
    backgroundColor: '#dc3545',
    transform: [{scale: 1.1}],
  },
  voiceButtonText: {
    fontSize: 18,
  },
  voiceStatusContainer: {
    backgroundColor: '#e3f2fd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    alignItems: 'center',
  },
  voiceStatusText: {
    color: '#1976d2',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
});

export default TranslationApp;