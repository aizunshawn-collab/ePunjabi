#!/usr/bin/env node

// Comprehensive Google Translate API checker
const https = require('https');

console.log('🔍 Google Translate API Activation Checker\n');

const API_KEY = 'AIzaSyBvq4sD8FsJkhDATyHk0pCQJsT_cy35xVc';

// Test 1: Check if API key works with language detection
async function testLanguageDetection() {
    return new Promise((resolve, reject) => {
        const url = `https://translation.googleapis.com/language/translate/v2/detect?key=${API_KEY}`;
        const postData = JSON.stringify({ q: 'Hello world' });

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
                resolve({ status: res.statusCode, data });
            });
        });

        req.on('error', reject);
        req.write(postData);
        req.end();
    });
}

// Test 2: Check supported languages
async function testSupportedLanguages() {
    return new Promise((resolve, reject) => {
        const url = `https://translation.googleapis.com/language/translate/v2/languages?key=${API_KEY}&target=en`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                resolve({ status: res.statusCode, data });
            });
        }).on('error', reject);
    });
}

// Test 3: Simple translation test
async function testTranslation() {
    return new Promise((resolve, reject) => {
        const url = `https://translation.googleapis.com/language/translate/v2?key=${API_KEY}`;
        const postData = JSON.stringify({
            q: 'Hello',
            source: 'en',
            target: 'pa',
            format: 'text'
        });

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
                resolve({ status: res.statusCode, data });
            });
        });

        req.on('error', reject);
        req.write(postData);
        req.end();
    });
}

async function runDiagnostics() {
    console.log('🚀 Running API diagnostics...\n');

    // Test 1: Language Detection
    try {
        console.log('📍 Test 1: Language Detection API...');
        const detectResult = await testLanguageDetection();
        console.log(`   Status: ${detectResult.status}`);
        
        if (detectResult.status === 200) {
            console.log('   ✅ Language Detection API: WORKING');
        } else {
            console.log('   ❌ Language Detection API: FAILED');
            console.log(`   Response: ${detectResult.data.substring(0, 200)}...`);
        }
    } catch (error) {
        console.log('   ❌ Language Detection API: ERROR -', error.message);
    }

    console.log('');

    // Test 2: Supported Languages
    try {
        console.log('📋 Test 2: Supported Languages API...');
        const langResult = await testSupportedLanguages();
        console.log(`   Status: ${langResult.status}`);
        
        if (langResult.status === 200) {
            console.log('   ✅ Supported Languages API: WORKING');
            const parsed = JSON.parse(langResult.data);
            if (parsed.data && parsed.data.languages) {
                console.log(`   📊 Found ${parsed.data.languages.length} supported languages`);
            }
        } else {
            console.log('   ❌ Supported Languages API: FAILED');
            console.log(`   Response: ${langResult.data.substring(0, 200)}...`);
        }
    } catch (error) {
        console.log('   ❌ Supported Languages API: ERROR -', error.message);
    }

    console.log('');

    // Test 3: Translation
    try {
        console.log('🔄 Test 3: Translation API...');
        const transResult = await testTranslation();
        console.log(`   Status: ${transResult.status}`);
        
        if (transResult.status === 200) {
            console.log('   ✅ Translation API: WORKING');
            const parsed = JSON.parse(transResult.data);
            if (parsed.data && parsed.data.translations) {
                console.log(`   🎯 Translation successful: "${parsed.data.translations[0].translatedText}"`);
            }
        } else {
            console.log('   ❌ Translation API: FAILED');
            const parsed = JSON.parse(transResult.data);
            if (parsed.error) {
                console.log(`   Error: ${parsed.error.message}`);
            }
        }
    } catch (error) {
        console.log('   ❌ Translation API: ERROR -', error.message);
    }

    console.log('\n' + '='.repeat(60));
    console.log('📋 ACTIVATION STATUS SUMMARY:');
    console.log('='.repeat(60));
    
    console.log('\n🔧 How to know if your API is fully activated:');
    console.log('   ✅ All tests should return status 200');
    console.log('   ✅ No error messages in responses');
    console.log('   ✅ Translation test should return Punjabi text');
    
    console.log('\n🛠️  If tests are failing, check:');
    console.log('   1. 🔑 API Key restrictions in Google Cloud Console');
    console.log('   2. 💳 Billing account is set up and active');
    console.log('   3. 🌐 API key has Translation API permissions');
    console.log('   4. ⏰ Wait a few more minutes for propagation');
    
    console.log('\n🔗 Useful links:');
    console.log('   • API Console: https://console.cloud.google.com/apis/api/translate.googleapis.com');
    console.log('   • Billing: https://console.cloud.google.com/billing');
    console.log('   • API Keys: https://console.cloud.google.com/apis/credentials');
}

runDiagnostics();