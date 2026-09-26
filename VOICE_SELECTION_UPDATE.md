# Voice Selection System Update

## Problem
Previously, all voice options (2 Punjabi + 4 English) were in a single radio button group. This meant selecting an English voice would deselect the Punjabi voice, and vice versa. This was problematic for bidirectional conversations where both languages are used.

## Solution
Separated the voice selection into two independent radio button groups:
- **Punjabi Voices**: 2 options (Female/Male)
- **English Voices**: 4 options (US Female/Male, UK Female/Male)

Now users can select one Punjabi voice AND one English voice simultaneously.

## Changes Made

### 1. HTML Structure (Lines 310-333)
- Changed Punjabi radio buttons: `name="punjabiVoice"`
- Changed English radio buttons: `name="englishVoice"`
- Each group maintains independent selection state

### 2. JavaScript Variables (Lines 427-428)
```javascript
let selectedPunjabiVoice = 'pa-IN-Wavenet-A'; // Default: Female Punjabi
let selectedEnglishVoice = 'en-US-Wavenet-F'; // Default: Female US English
```

### 3. Event Listeners (Lines 538-553)
- Separate listener for `punjabiVoice` radio group
- Separate listener for `englishVoice` radio group
- Each updates its respective variable and shows status message

### 4. UK/US Toggle Function (Lines 437-462)
- Updated to only modify `selectedEnglishVoice`
- No longer affects Punjabi voice selection
- Switches between US and UK English voices

### 5. Text-to-Speech Function (Lines 777-809)
**New Signature:**
```javascript
async function speakWithGoogleTTS(text, apiKeyToUse = null, languageHint = null)
```

**Language Detection:**
- Detects Punjabi characters using regex: `/[\u0A00-\u0A7F]/`
- Uses `languageHint` parameter ('pa' or 'en') if provided
- Automatically selects appropriate voice:
  - Punjabi text → uses `selectedPunjabiVoice`
  - English text → uses `selectedEnglishVoice`

### 6. Translation Functions (Lines 1391-1411, 1537-1558)
Updated both translation workflows to pass language hints:
- **Punjabi → English**: `await speakWithGoogleTTS(translatedText, null, 'en');`
- **English → Punjabi**: `await speakWithGoogleTTS(translatedText, null, 'pa');`

Also replaced browser TTS with Google TTS for English translations to use premium voices.

## User Experience
1. User selects preferred Punjabi voice (Female/Male)
2. User selects preferred English voice and accent (US/UK, Female/Male)
3. During translation conversations:
   - English input → Punjabi output uses selected Punjabi voice
   - Punjabi input → English output uses selected English voice
4. Both selections remain active throughout the session
5. UK/US toggle only affects English voice selection

## Benefits
- ✅ Independent voice selection per language
- ✅ Consistent voices throughout bidirectional conversations
- ✅ Natural conversation flow between English and Punjabi speakers
- ✅ Respects user preferences for both languages simultaneously
- ✅ Premium Google Wavenet voices for both languages
