/**
 * BoundingBoxOverlay.tsx
 * Pure-View bounding-box renderer. Cheap and dependency-free — if you want
 * smoother shared-value-driven boxes later, swap to react-native-skia.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Detection } from './postprocess';

const COLORS: Record<string, string> = {
  camo: '#22c55e',
  gun: '#ef4444',
};

export default function BoundingBoxOverlay({ detections }: { detections: Detection[] }) {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {detections.map((d, i) => {
        const color = COLORS[d.className] ?? '#3b82f6';
        return (
          <View
            key={i}
            style={[
              styles.box,
              {
                left: d.x,
                top: d.y,
                width: d.w,
                height: d.h,
                borderColor: color,
              },
            ]}
          >
            <View style={[styles.label, { backgroundColor: color }]}>
              <Text style={styles.labelText}>
                {d.className} {(d.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  label: {
    position: 'absolute',
    left: -2,
    top: -22,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
  },
  labelText: { color: 'black', fontWeight: '700', fontSize: 11 },
});
