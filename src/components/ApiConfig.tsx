import React, {useState} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
} from 'react-native';
import TranslationService from '../services/TranslationService';

interface ApiConfigProps {
  onApiKeySet: () => void;
}

const ApiConfig: React.FC<ApiConfigProps> = ({onApiKeySet}) => {
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSetApiKey = async () => {
    if (!apiKey.trim()) {
      Alert.alert('Error', 'Please enter a valid API key');
      return;
    }

    setIsLoading(true);
    try {
      await TranslationService.setApiKey(apiKey.trim());
      Alert.alert('Success', 'API key has been saved successfully!', [
        {text: 'OK', onPress: onApiKeySet},
      ]);
    } catch (error) {
      Alert.alert('Error', 'Failed to save API key. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Setup Google Translate API</Text>
      
      <Text style={styles.description}>
        To use this translation app, you need to configure your Google Cloud Translation API key.
      </Text>

      <View style={styles.stepsContainer}>
        <Text style={styles.stepsTitle}>Setup Instructions:</Text>
        
        <Text style={styles.step}>
          1. Go to{' '}
          <Text style={styles.link}>Google Cloud Console</Text>
        </Text>
        
        <Text style={styles.step}>
          2. Create a new project or select an existing one
        </Text>
        
        <Text style={styles.step}>
          3. Enable the Cloud Translation API
        </Text>
        
        <Text style={styles.step}>
          4. Create credentials (API key)
        </Text>
        
        <Text style={styles.step}>
          5. Copy your API key and paste it below
        </Text>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Google Translate API Key:</Text>
        <TextInput
          style={styles.input}
          value={apiKey}
          onChangeText={setApiKey}
          placeholder="Enter your API key here..."
          secureTextEntry
          multiline={false}
        />
      </View>

      <TouchableOpacity
        style={[styles.button, isLoading && styles.buttonDisabled]}
        onPress={handleSetApiKey}
        disabled={isLoading}>
        <Text style={styles.buttonText}>
          {isLoading ? 'Saving...' : 'Save API Key'}
        </Text>
      </TouchableOpacity>

      <Text style={styles.note}>
        Note: Your API key is stored securely on your device and is only used for translation requests.
      </Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  description: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 30,
    color: '#666',
    lineHeight: 24,
  },
  stepsContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 20,
    marginBottom: 30,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  stepsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#333',
  },
  step: {
    fontSize: 14,
    marginBottom: 10,
    color: '#555',
    lineHeight: 20,
  },
  link: {
    color: '#007AFF',
    textDecorationLine: 'underline',
  },
  inputContainer: {
    marginBottom: 30,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 25,
    paddingVertical: 15,
    paddingHorizontal: 30,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  note: {
    fontSize: 12,
    textAlign: 'center',
    color: '#888',
    fontStyle: 'italic',
    lineHeight: 18,
  },
});

export default ApiConfig;