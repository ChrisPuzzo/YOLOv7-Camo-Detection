// Learn more https://docs.expo.dev/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Treat model files as bundleable static assets, not JS modules.
// Required so `require('../assets/models/camo_int8.tflite')` resolves
// during EAS Build.
config.resolver.assetExts.push('tflite', 'pt', 'onnx', 'bin');

module.exports = config;
