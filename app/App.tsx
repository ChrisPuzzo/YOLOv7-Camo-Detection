import React from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import CameraScreen from './src/CameraScreen';
import { ErrorBoundary } from './src/ErrorScreen';

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar style="light" />
      <ErrorBoundary>
        <CameraScreen />
      </ErrorBoundary>
    </GestureHandlerRootView>
  );
}
