# English-Punjabi Voice Translation App

A React Native Android application for translating between English and Punjabi using Google Translate API, built with AndroidX support.

## Features

- 🔄 **Bidirectional Translation**: Translate from English to Punjabi and vice versa
- 🎯 **Simple Interface**: Clean, intuitive UI with language swap functionality
- 🔑 **Secure API Integration**: Google Cloud Translation API with secure key storage
- 📱 **Android Optimized**: Built with AndroidX for modern Android development
- ⚡ **Fast Performance**: Efficient translation with error handling

## Screenshots

*Note: Screenshots will be available after building and running the app*

## Prerequisites

Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (v16 or later)
- [React Native CLI](https://reactnative.dev/docs/environment-setup)
- [Android Studio](https://developer.android.com/studio) with Android SDK
- [Java Development Kit (JDK)](https://www.oracle.com/java/technologies/downloads/) (v11 or later)
- [Git](https://git-scm.com/)

## Google Cloud Setup

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Cloud Translation API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Translation API"
   - Click "Enable"

3. **Create API Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy your API key (keep it secure!)

4. **Optional - Restrict API Key**:
   - Click on your API key to edit
   - Under "API restrictions", select "Restrict key"
   - Choose "Cloud Translation API"

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd EnglishPunjabiVoiceTranslationApp
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Android Setup**:
   ```bash
   cd android
   ./gradlew clean
   cd ..
   ```

## Configuration

1. **API Key Setup**:
   - Copy `.env.example` to `.env.local`
   - Add your Google Translate API key:
     ```
     GOOGLE_TRANSLATE_API_KEY=your_actual_api_key_here
     ```

2. **Android Device/Emulator**:
   - Connect an Android device with USB debugging enabled, OR
   - Start an Android emulator from Android Studio

## Running the App

### Method 1: Using VS Code Tasks

1. Open the project in VS Code
2. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
3. Type "Tasks: Run Task"
4. Select "React Native: Run Android"

### Method 2: Using Terminal Commands

1. **Start Metro bundler**:
   ```bash
   npm start
   ```

2. **In a new terminal, run Android**:
   ```bash
   npm run android
   ```

### Method 3: Manual Build

1. **Build the Android app**:
   ```bash
   cd android
   ./gradlew assembleDebug
   cd ..
   ```

2. **Install on device**:
   ```bash
   npx react-native run-android
   ```

## Project Structure

```
EnglishPunjabiVoiceTranslationApp/
├── android/                    # Android-specific code and build files
│   ├── app/
│   │   ├── build.gradle       # App-level build configuration
│   │   └── src/main/
│   │       └── AndroidManifest.xml
│   ├── build.gradle           # Project-level build configuration
│   └── gradle.properties      # AndroidX configuration
├── src/                       # React Native source code
│   ├── components/
│   │   └── ApiConfig.tsx      # API key configuration component
│   ├── services/
│   │   └── TranslationService.ts  # Google Translate API integration
│   └── TranslationApp.tsx     # Main app component
├── .github/
│   └── copilot-instructions.md # Development guidelines
├── .vscode/
│   └── tasks.json             # VS Code tasks for build and run
├── App.tsx                    # App entry point
├── index.js                   # React Native entry point
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── babel.config.js            # Babel configuration
├── metro.config.js            # Metro bundler configuration
└── README.md                  # This file
```

## Available Scripts

- `npm start` - Start Metro bundler
- `npm run android` - Run on Android device/emulator
- `npm run lint` - Run ESLint code checking
- `npm test` - Run Jest tests
- `npm install` - Install dependencies

## API Integration

The app uses Google Cloud Translation API for text translation:

- **Endpoint**: `https://translation.googleapis.com/language/translate/v2`
- **Supported Languages**: English (en) and Punjabi (pa)
- **Features**: Translation, language detection, error handling
- **Security**: API key stored securely using AsyncStorage

## AndroidX Support

This project uses AndroidX instead of the older Android Support Library:

- `android.useAndroidX=true` in `gradle.properties`
- `android.enableJetifier=true` for automatic dependency migration
- Modern Android components and APIs

## Troubleshooting

### Common Issues

1. **Metro bundler issues**:
   ```bash
   npx react-native start --reset-cache
   ```

2. **Android build fails**:
   ```bash
   cd android
   ./gradlew clean
   cd ..
   npm run android
   ```

3. **API key not working**:
   - Verify the API key is correct
   - Ensure Cloud Translation API is enabled
   - Check API quotas and billing

4. **AndroidX issues**:
   ```bash
   npm run postinstall
   ```

### Debug Mode

- Shake device or press `Ctrl+M` (emulator) / `Cmd+M` (iOS) to open debug menu
- Enable "Fast Refresh" for development
- Use React Developer Tools in Chrome

## Development

### Code Style

- ESLint configuration included
- Prettier formatting recommended
- TypeScript for type safety

### Testing

```bash
npm test
```

### Building for Production

1. **Generate signed APK**:
   ```bash
   cd android
   ./gradlew assembleRelease
   ```

2. **Find APK at**:
   ```
   android/app/build/outputs/apk/release/app-release.apk
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review [React Native documentation](https://reactnative.dev/docs/getting-started)
3. Check [Google Cloud Translation API docs](https://cloud.google.com/translate/docs)
4. Create an issue in this repository

## Acknowledgments

- React Native team for the excellent framework
- Google Cloud for the Translation API
- AndroidX team for modern Android development tools

---

**Happy Translating! 🎉**