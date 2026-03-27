const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// Serve static files
app.use(express.static('.'));

// Translation proxy endpoint
app.post('/api/translate', async (req, res) => {
    const API_KEY = 'AIzaSyBvq4sD8FsJkhDATyHk0pCQJsT_cy35xVc';
    const API_URL = 'https://translation.googleapis.com/language/translate/v2';
    
    try {
        console.log('Translation request:', req.body);
        
        // Handle different request formats - normalize to Google Translate API format
        let translateRequest;
        if (req.body.text && req.body.sourceLang && req.body.targetLang) {
            // Voice app format: { text, sourceLang, targetLang }
            translateRequest = {
                q: req.body.text,
                source: req.body.sourceLang,
                target: req.body.targetLang,
                format: 'text'
            };
        } else {
            // Direct Google Translate format: { q, source, target }
            translateRequest = req.body;
        }
        
        console.log('Normalized request:', translateRequest);
        
        // Use dynamic import for fetch in Node.js environments that don't have it built-in
        let fetchFn;
        try {
            fetchFn = fetch; // Try built-in fetch first (Node 18+)
        } catch {
            const { default: fetch } = await import('node-fetch');
            fetchFn = fetch;
        }
        
        const response = await fetchFn(`${API_URL}?key=${API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(translateRequest)
        });
        
        const data = await response.json();
        console.log('Translation response:', data);
        
        if (response.ok) {
            // Check if this is a voice app request and format response accordingly
            if (req.body.text && req.body.sourceLang && req.body.targetLang) {
                // Voice app expects: { translatedText: "..." }
                const translatedText = data.data.translations[0].translatedText;
                res.json({ translatedText });
            } else {
                // Return original Google Translate format
                res.json(data);
            }
        } else {
            console.error('Translation API error:', data);
            res.status(response.status).json(data);
        }
    } catch (error) {
        console.error('Translation proxy error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Translation proxy server is running' });
});

app.listen(PORT, () => {
    console.log(`🚀 Translation server running at http://localhost:${PORT}`);
    console.log(`📝 Access web demo at: http://localhost:${PORT}/web-demo.html`);
    console.log(`🔧 Access debug test at: http://localhost:${PORT}/debug-test.html`);
});