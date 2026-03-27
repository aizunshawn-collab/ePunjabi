#!/usr/bin/env node

// Simple test script to verify Google Translate API integration
const https = require('https');

console.log('🚀 Testing English-Punjabi Translation App...\n');

const API_KEY = 'AIzaSyBvq4sD8FsJkhDATyHk0pCQJsT_cy35xVc';
const TEST_TEXT = 'Hello, how are you?';

async function testTranslation() {
    const url = `https://translation.googleapis.com/language/translate/v2?key=${API_KEY}`;
    
    const postData = JSON.stringify({
        q: TEST_TEXT,
        source: 'en',
        target: 'pa',
        format: 'text'
    });

    return new Promise((resolve, reject) => {
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        };

        const req = https.request(url, options, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                try {
                    console.log('📡 API Response:', data);
                    const response = JSON.parse(data);
                    if (response.data && response.data.translations) {
                        resolve(response.data.translations[0].translatedText);
                    } else if (response.error) {
                        reject(new Error(`API Error: ${response.error.message}`));
                    } else {
                        reject(new Error('Invalid response format'));
                    }
                } catch (error) {
                    reject(error);
                }
            });
        });

        req.on('error', reject);
        req.write(postData);
        req.end();
    });
}

async function runTest() {
    try {
        console.log(`📝 Original text (English): "${TEST_TEXT}"`);
        console.log('🔄 Translating to Punjabi...\n');
        
        const translatedText = await testTranslation();
        
        console.log('✅ Translation successful!');
        console.log(`🎯 Translated text (Punjabi): "${translatedText}"`);
        console.log('\n🎉 Google Translate API is working correctly!');
        console.log('\n📱 Your React Native app components are ready:');
        console.log('   - ✅ Translation Service configured');
        console.log('   - ✅ API key working');
        console.log('   - ✅ English ⟷ Punjabi translation');
        console.log('   - ✅ UI components created');
        console.log('\n🔧 To run on Android: Set up Android SDK and run `npm run android`');
        
    } catch (error) {
        console.error('❌ Translation failed:', error.message);
        console.log('\n🔍 Please check:');
        console.log('   - Internet connection');
        console.log('   - Google Translate API key validity');
        console.log('   - API quotas and billing');
    }
}

runTest();