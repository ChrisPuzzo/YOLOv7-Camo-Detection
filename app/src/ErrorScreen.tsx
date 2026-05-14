/**
 * ErrorScreen.tsx
 * ---------------
 * Friendly full-screen error UI used in two places:
 *   1. Top-level <ErrorBoundary> below — catches React render errors anywhere
 *      in the app so we never show a white "RN Red Box" to a beta tester.
 *   2. The model-load failure branch inside CameraScreen.
 *
 * Surfaces the raw error message + a couple of common-cause hints, and shows
 * a "Try again" button that re-mounts the children (or fires a custom retry).
 */
import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Platform,
} from 'react-native';

type Props = {
  title?: string;
  message: string;
  hint?: string;
  details?: string;
  onRetry?: () => void;
};

export default function ErrorScreen({
  title = 'Something went wrong',
  message,
  hint,
  details,
  onRetry,
}: Props) {
  return (
    <View style={styles.root}>
      <Text style={styles.icon}>⚠️</Text>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.message}>{message}</Text>

      {hint ? <Text style={styles.hint}>{hint}</Text> : null}

      {details ? (
        <ScrollView style={styles.details} contentContainerStyle={{ padding: 12 }}>
          <Text style={styles.detailsHeading}>Technical details</Text>
          <Text style={styles.detailsText} selectable>
            {details}
          </Text>
        </ScrollView>
      ) : null}

      {onRetry ? (
        <TouchableOpacity style={styles.btn} onPress={onRetry}>
          <Text style={styles.btnText}>Try again</Text>
        </TouchableOpacity>
      ) : null}

      <Text style={styles.footer}>
        Camo Detector Beta · {Platform.OS} · v0.1.0
      </Text>
    </View>
  );
}

/**
 * Top-level error boundary. Any uncaught render error in any descendant
 * shows the ErrorScreen instead of crashing the app.
 *
 * Worklet-thread crashes (frame processor, TFLite invoke) bubble up via the
 * vision-camera onError callback — we handle those at the CameraScreen layer.
 */
type BoundaryState = { error: Error | null };
export class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  BoundaryState
> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): BoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Log to console so `adb logcat | grep ReactNativeJS` picks it up.
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary]', error, info?.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      const e = this.state.error;
      return (
        <ErrorScreen
          title="App crashed"
          message={e.message || 'An unexpected error occurred.'}
          hint="This usually clears on retry. If it keeps happening, file an issue with the technical details below."
          details={`${e.name}: ${e.message}\n\n${e.stack ?? '(no stack)'}`}
          onRetry={this.reset}
        />
      );
    }
    return this.props.children;
  }
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#141C12',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 48,
  },
  icon: { fontSize: 56, marginBottom: 12 },
  title: {
    color: 'white',
    fontSize: 22,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 12,
  },
  message: {
    color: '#e2e8f0',
    fontSize: 15,
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 22,
  },
  hint: {
    color: '#a3b18a',
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 18,
  },
  details: {
    maxHeight: 220,
    width: '100%',
    backgroundColor: 'rgba(0,0,0,0.4)',
    borderRadius: 8,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  detailsHeading: {
    color: '#94a3b8',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    marginBottom: 6,
    textTransform: 'uppercase',
  },
  detailsText: {
    color: '#cbd5e1',
    fontSize: 11,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
    lineHeight: 16,
  },
  btn: {
    paddingHorizontal: 28,
    paddingVertical: 12,
    backgroundColor: '#a3b18a',
    borderRadius: 10,
  },
  btnText: { color: '#141C12', fontWeight: '700', fontSize: 15 },
  footer: {
    position: 'absolute',
    bottom: 16,
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
  },
});
