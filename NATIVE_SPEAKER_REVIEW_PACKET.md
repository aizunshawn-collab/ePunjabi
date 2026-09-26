# Native Speaker Review Packet

Generated automatically from the live backend. For each row, English/Punjabi source
is on top, the app's actual current translation is below it.


## What to check for each pair
For every row below, please mark:
1. **Correct meaning?** (yes / no / partially - describe what's wrong)
2. **Natural phrasing?** (would a native speaker actually say it this way, or does it sound
   stiff/translated/robotic?)
3. **Tone/formality** appropriate for the context (casual vs. respectful/formal)?
4. **Gender agreement** correct throughout (especially section F)?

Sections C, D, and E are KNOWN issues already found by automated testing - your read on
*how bad* they are (embarrassing vs. minor vs. actually fine in context) is the most
valuable feedback, since automated tests can't judge that.

## A. Everyday conversation (baseline sanity check)

**1. (en -> pa)**
- Source: Good morning, how did you sleep?
- Translation: ਚੰਗੀ ਸਵੇਰ, ਤੁਸੀਂ ਕਿਵੇਂ ਸੌਂ ਗਏ?
- App's own confidence score: 1.00

**2. (en -> pa)**
- Source: Can you please help me find my way to the train station?
- Translation: ਕੀ ਤੁਸੀਂ ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਤੱਕ ਪਹੁੰਚਣ ਦਾ ਰਸਤਾ ਲੱਭਣ ਵਿੱਚ ਮੇਰੀ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?
- App's own confidence score: 0.92

**3. (en -> pa)**
- Source: I would like to book a room for two nights.
- Translation: ਮੈਂ ਦੋ ਰਾਤਾਂ ਲਈ ਕਮਰਾ ਬੁੱਕ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ।
- App's own confidence score: 0.73

**4. (pa -> en)**
- Source: ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ, ਇਸ ਲਈ ਚਲੋ ਸੈਰ ਲਈ ਚੱਲੀਏ।
- Translation: The weather is great today, so let's go for a walk.
- App's own confidence score: 0.84

**5. (pa -> en)**
- Source: ਮੌਸਮ ਵਿਭਾਗ ਨੇ ਕੱਲ੍ਹ ਮੀਂਹ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਹੈ।
- Translation: The Meteorological Department has predicted rain tomorrow.
- App's own confidence score: 0.93


## B. Travel / medical / emergency (should be exact - glossary-locked)

**1. (en -> pa)**
- Source: I need a doctor, call an ambulance please.
- Translation: ਮੈਨੂੰ ਡਾਕਟਰ ਦੀ ਜ਼ਰੂਰਤ ਹੈ, ਕਿਰਪਾ ਕਰਕੇ ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ।
- App's own confidence score: 1.00

**2. (en -> pa)**
- Source: I lost my passport at the airport.
- Translation: ਮੇਰਾ ਪਾਸਪੋਰਟ ਹਵਾਈ ਅੱਡੇ ਉੱਤੇ ਗੁੰਮ ਹੋ ਗਿਆ।
- App's own confidence score: 1.00

**3. (pa -> en)**
- Source: ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ ਅਤੇ ਸਿਰ ਬਹੁਤ ਦਰਦ ਕਰਦਾ ਹੈ।
- Translation: I have a fever and a severe headache.
- App's own confidence score: 0.64


## C. KNOWN BUG - alphanumeric codes (expected WRONG, please describe how bad)

**1. (en -> pa)**
- Source: Your flight number is AI 202, and the booking reference is ABX123.
- Translation: ਤੁਹਾਡਾ ਫਲਾਈਟ ਨੰਬਰ AI 202 ਹੈ, ਅਤੇ ਬੁਕਿੰਗ ਸੰਦਰਭ ABX123 ਹੈ।
- App's own confidence score: 1.00
- ⚠️ Note: Expected bug: codes get phonetically spelled out in Gurmukhi instead of kept as-is.

**2. (en -> pa)**
- Source: Please quote invoice number INV-4567 when you call.
- Translation: ਜਦੋਂ ਤੁਸੀਂ ਕਾਲ ਕਰਦੇ ਹੋ ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਇਨਵੁਆਇਸ ਨੰਬਰ ਆਈ. ਐੱਨ. ਵੀ.-4567 ਦਾ ਹਵਾਲਾ ਦਿਓ।
- App's own confidence score: 0.54
- ⚠️ Note: Expected bug: same as above.


## D. KNOWN BUG - proper noun handling (mixed: one wrong, one right)

**1. (en -> pa)**
- Source: My name is Harpreet Singh and I live in Amritsar.
- Translation: ਮੇਰਾ ਨਾਮ ਗੁਰਪ੍ਰੀਤ ਸਿੰਘ ਹੈ ਅਤੇ ਮੈਂ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਰਹਿੰਦਾ ਹਾਂ।
- App's own confidence score: 0.82
- ⚠️ Note: Expected bug: model has previously substituted a different name (Gurpreet) here.

**2. (en -> pa)**
- Source: My name is Jaspreet Kaur and I live in Ludhiana.
- Translation: ਮੇਰਾ ਨਾਮ ਜਸਪ੍ਰਿਤ ਕੌਰ ਹੈ ਅਤੇ ਮੈਂ ਲੁਧਿਆਣੇ ਵਿੱਚ ਰਹਿੰਦੀ ਹਾਂ।
- App's own confidence score: 1.00
- ⚠️ Note: Expected to be correct - included as a working comparison case.


## E. KNOWN EDGE CASE - narrow admission/ਦਾਖਲਾ drop

**1. (pa -> en)**
- Source: ਮੇਰੇ ਬੇਟੇ ਦਾ ਸਕੂਲ ਵਿੱਚ ਦਾਖਲਾ ਅਗਲੇ ਹਫ਼ਤੇ ਹੋਵੇਗਾ।
- Translation: My son will be in school next week.
- App's own confidence score: 0.50
- ⚠️ Note: Expected bug: 'admission/enrollment' meaning has been dropped in this exact phrasing before.

**2. (pa -> en)**
- Source: ਕਾਲਜ ਵਿੱਚ ਦਾਖਲੇ ਦੀ ਆਖਰੀ ਤਾਰੀਖ ਕੀ ਹੈ?
- Translation: What is the last date of admission to the college?
- App's own confidence score: 0.75
- ⚠️ Note: Expected to be correct - included as a working comparison case.


## F. Long/complex sentence (recently fixed chunking bug - please confirm still holds)

**1. (en -> pa)**
- Source: In order to apply for the small business loan, you will need to submit proof of residence, a valid business license, and a detailed plan describing how you intend to use the funds over the next twelve months.
- Translation: ਛੋਟੇ ਕਾਰੋਬਾਰੀ ਕਰਜ਼ੇ ਲਈ ਅਰਜ਼ੀ ਦੇਣ ਲਈ, ਤੁਹਾਨੂੰ ਨਿਵਾਸ ਦਾ ਸਬੂਤ, ਇੱਕ ਜਾਇਜ਼ ਵਪਾਰਕ ਲਾਇਸੈਂਸ ਅਤੇ ਇੱک ਵਿਸਤ੍ਰਿਤ ਯੋਜਨਾ ਪੇਸ਼ ਕਰਨ ਦੀ ਜ਼ਰੂਰਤ ਹੋਏਗੀ ਜਿਸ ਵਿੱਚ ਦੱਸਿਆ ਗਿਆ ਹੈ ਕਿ ਤੁਸੀਂ ਅਗਲੇ ਬਾਰਾਂ ਮਹੀਨਿਆਂ ਵਿੱਚੋਂ ਫੰਡਾਂ ਦੀ ਵਰਤੋਂ ਕਿਵੇਂ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ।
- App's own confidence score: 0.76
- ⚠️ Note: Previously this sentence dropped the first two conditions entirely when split into chunks.

**2. (pa -> en)**
- Source: ਮੇਰੀ ਮਾਂ ਹਰ ਰੋਜ਼ ਸਵੇਰੇ ਜਲਦੀ ਉੱਠਦੀ ਹੈ, ਨਾਸ਼ਤਾ ਬਣਾਉਂਦੀ ਹੈ, ਅਤੇ ਫਿਰ ਸ਼ਾਮ ਨੂੰ ਕੰਮ ਤੋਂ ਬਾਅਦ ਰਾਤ ਦੀਆਂ ਕਲਾਸਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਜਾਂਦੀ ਹੈ ਕਿਉਂਕਿ ਉਹ ਨਰਸ ਬਣਨਾ ਚਾਹੁੰਦੀ ਹੈ।
- Translation: My mother wakes up early every morning, makes breakfast, and then goes to night classes after work in the evening because she wants to be a nurse.
- App's own confidence score: 0.81
- ⚠️ Note: Previously this sentence lost the subject's gender ('she' became 'he') partway through.


## G. Code-switching (Punjabi-English mixed - confirmed working, spot-check)

**1. (pa -> en)**
- Source: ਮੈਨੂੰ ਅੱਜ office ਵਿੱਚ ਦੇਰ ਹੋ ਜਾਵੇਗੀ।
- Translation: I'll be late for the office today.
- App's own confidence score: 0.53

**2. (pa -> en)**
- Source: ਉਹ ਮੇਰਾ best friend ਹੈ।
- Translation: He's my best friend.
- App's own confidence score: 0.33

