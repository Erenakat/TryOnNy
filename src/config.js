import { NativeModules, Platform } from 'react-native';
import Constants from 'expo-constants';

// AI-backend URL.
// - Android emulator: 10.0.2.2
// - iOS simulator: localhost
// - Physical device: use the dev server host IP (derived from Expo hostUri/debuggerHost)
const FALLBACK_DEV_BACKEND_HOST = 'http://10.0.0.27:8000'; // fallback hvis auto-detektering feiler

function getExpoHostIp() {
  const hostUri =
    Constants?.expoConfig?.hostUri ||
    Constants?.manifest2?.extra?.expoClient?.hostUri ||
    Constants?.manifest?.debuggerHost ||
    Constants?.expoGoConfig?.debuggerHost ||
    Constants?.debuggerHost ||
    null;

  if (!hostUri || typeof hostUri !== 'string') return null;
  // hostUri/debuggerHost looks like "192.168.1.10:19000" or "192.168.1.10:19000,192.168.1.10:19001"
  const first = hostUri.split(',')[0]?.trim();
  const ip = first?.split(':')[0]?.trim();
  return ip || null;
}

function getPackagerHost() {
  const scriptURL = NativeModules?.SourceCode?.scriptURL;
  if (scriptURL && typeof scriptURL === 'string') {
    // Typical: http://192.168.1.10:8081/index.bundle?... OR exp://192.168.1.10:8081
    const m = scriptURL.match(/^[a-z]+:\/\/([^/:]+)(?::\d+)?/i);
    if (m?.[1]) return m[1];
  }
  return getExpoHostIp();
}

function getDevBackendHost() {
  const env = process.env.EXPO_PUBLIC_AI_BACKEND_URL;
  if (env) return env;

  const host = getPackagerHost();
  if (host) {
    // If host is localhost, map Android emulator to 10.0.2.2. Otherwise use the host directly.
    const isLocal = host === 'localhost' || host === '127.0.0.1';
    if (Platform.OS === 'android' && isLocal) return 'http://10.0.2.2:8000';
    if (Platform.OS === 'ios' && isLocal) return 'http://localhost:8000';
    return `http://${host}:8000`;
  }

  return FALLBACK_DEV_BACKEND_HOST;
}

const AI_BACKEND_URL = __DEV__ ? getDevBackendHost() : (process.env.EXPO_PUBLIC_AI_BACKEND_URL || 'https://din-ai-backend.up.railway.app');

export { AI_BACKEND_URL };
