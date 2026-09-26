import API_CONFIG from '../config/apiConfig';

// Simple test to verify API key configuration
export const testApiConfiguration = () => {
  console.log('API Key configured:', !!API_CONFIG.GOOGLE_TRANSLATE_API_KEY);
  console.log('API Key length:', API_CONFIG.GOOGLE_TRANSLATE_API_KEY?.length || 0);
  
  return {
    isConfigured: !!API_CONFIG.GOOGLE_TRANSLATE_API_KEY,
    keyLength: API_CONFIG.GOOGLE_TRANSLATE_API_KEY?.length || 0,
  };
};

export default testApiConfiguration;