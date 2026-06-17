import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  StyleSheet,
  Text,
  View,
  Image,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  AppState,
  Linking,
  Modal,
  Pressable,
} from 'react-native';
import Slider from '@react-native-community/slider';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Path, Circle } from 'react-native-svg';
import { AI_BACKEND_URL } from './src/config';
import GLBViewer from './src/GLBViewer';

const AVATAR_KEY = '@tryon_avatar';
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 3 * 60 * 1000;
const SCREEN_NAME = 'ProfileAvatar';
const ERROR_MESSAGES = {
  POSE_DEPENDENCY_MISSING: 'Server mangler pose-komponent (mediapipe/opencv).',
  FEATURE_DISABLED: 'Kroppsanalyse er deaktivert.',
  POSE_MODEL_INIT_FAILED: 'Kroppsanalyse kunne ikke initialiseres på serveren.',
  POSE_TIMEOUT: 'Serveren er opptatt, prøv igjen.',
  POSE_OOM: 'Serveren er opptatt, prøv igjen.',
  POSE_IMAGE_DECODE_FAILED: 'Kunne ikke lese kroppsbildet. Prøv et annet bilde.',
  JOB_CREATE_FAILED: 'Kunne ikke starte generering på serveren. Prøv igjen.',
  INPUT_INVALID: 'Vennligst last opp full-body bilde.',
};

const resolveBackendError = (payload) => {
  if (!payload) return { message: 'Generering feilet', errorCode: null, requestId: null };
  const errorCode = payload.error_code || null;
  const requestId = payload.request_id || null;
  const details = payload.details || null;
  const fallbackMessage = payload.message || payload.error || payload.detail || 'Generering feilet';
  const message = errorCode && ERROR_MESSAGES[errorCode] ? ERROR_MESSAGES[errorCode] : fallbackMessage;
  return { message, errorCode, requestId, details };
};

const normalizeHex = (hex) => {
  if (!hex) return null;
  const s = String(hex).trim();
  if (!s) return null;
  return s.startsWith('#') ? s.toUpperCase() : `#${s.toUpperCase()}`;
};

const polarToCartesian = (cx, cy, r, angleDeg) => {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
};

const describeDonutSegment = (cx, cy, rOuter, rInner, startAngle, endAngle) => {
  const startOuter = polarToCartesian(cx, cy, rOuter, startAngle);
  const endOuter = polarToCartesian(cx, cy, rOuter, endAngle);
  const startInner = polarToCartesian(cx, cy, rInner, endAngle);
  const endInner = polarToCartesian(cx, cy, rInner, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1';
  return [
    `M ${startOuter.x} ${startOuter.y}`,
    `A ${rOuter} ${rOuter} 0 ${largeArcFlag} 1 ${endOuter.x} ${endOuter.y}`,
    `L ${startInner.x} ${startInner.y}`,
    `A ${rInner} ${rInner} 0 ${largeArcFlag} 0 ${endInner.x} ${endInner.y}`,
    'Z',
  ].join(' ');
};

const ColorWheel = ({ colors, size = 240, ringWidth = 56 }) => {
  const palette = Array.isArray(colors) ? colors.filter(Boolean) : [];
  const n = Math.max(6, Math.min(palette.length || 0, 12)) || 12;
  const filled = [];
  for (let i = 0; i < n; i += 1) filled.push(palette[i] || '#E6E8EF');

  const cx = size / 2;
  const cy = size / 2;
  const rOuter = size / 2;
  const rInner = Math.max(0, rOuter - ringWidth);
  const step = 360 / filled.length;

  return (
    <View style={styles.paletteWheelWrap}>
      <Svg width={size} height={size}>
        {filled.map((hex, idx) => {
          const start = idx * step;
          const end = start + step;
          const d = describeDonutSegment(cx, cy, rOuter, rInner, start, end);
          return <Path key={`seg-${idx}`} d={d} fill={hex} />;
        })}
        <Circle cx={cx} cy={cy} r={rInner} fill="#fff" />
      </Svg>
    </View>
  );
};

export default function App() {
  const [faceImage, setFaceImage] = useState(null);
  const [bodyFrontImage, setBodyFrontImage] = useState(null);
  const [bodySideImage, setBodySideImage] = useState(null);
  const [avatarUrl, setAvatarUrl] = useState(null);
  const [aiAvatar, setAiAvatar] = useState(null);
  const [avatarStyle, setAvatarStyle] = useState('female'); // 'male' | 'female' | 'neutral'
  const [activeTab, setActiveTab] = useState('avatar'); // 'avatar' | 'color'
  const [rotation, setRotation] = useState(0);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobProgressMessage, setJobProgressMessage] = useState('');
  const [jobError, setJobError] = useState(null);
  const [jobErrorCode, setJobErrorCode] = useState(null);
  const [jobRequestId, setJobRequestId] = useState(null);
  const [jobErrorDetails, setJobErrorDetails] = useState(null);
  const [bodyDebug, setBodyDebug] = useState(null);
  const [pickerNotice, setPickerNotice] = useState(null);
  const [showFaceLikenessDebug, setShowFaceLikenessDebug] = useState(false);
  const [permissionModal, setPermissionModal] = useState({ visible: false, kind: 'library', message: '' });
  const [colorAnalysisResult, setColorAnalysisResult] = useState(null);
  const [colorAnalysisLoading, setColorAnalysisLoading] = useState(false);
  const [colorAnalysisError, setColorAnalysisError] = useState(null);
  const pollTimeoutRef = useRef(null);
  const pickingRef = useRef(false);
  const pickerSessionRef = useRef(0);
  const pickerCooldownUntilRef = useRef(0);
  const lastPickerCloseAtRef = useRef(0);
  const lastPickerLabelRef = useRef(null);
  const appStateRef = useRef(AppState.currentState);
  const [isPickingImage, setIsPickingImage] = useState(false);

  const logDev = (event, payload = null) => {
    if (!__DEV__) return;
    if (payload === null || payload === undefined) {
      console.log(`[${SCREEN_NAME}] ${event}`);
      return;
    }
    console.log(`[${SCREEN_NAME}] ${event}`, payload);
  };

  useEffect(() => {
    logDev('screen mount');
    loadAvatar();
    return () => {
      if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
      logDev('screen unmount');
    };
  }, []);

  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      const prevState = appStateRef.current;
      appStateRef.current = nextState;
      logDev('appState change', { prevState, nextState });
      if (prevState !== 'active' && nextState === 'active') {
        logDev('focus event', { screen: SCREEN_NAME });
        if (Date.now() - lastPickerCloseAtRef.current < 4000 && lastPickerLabelRef.current) {
          logDev('focus after picker close', { label: lastPickerLabelRef.current });
        }
      } else if (prevState === 'active' && nextState !== 'active') {
        logDev('blur event', { screen: SCREEN_NAME });
      }
    });
    return () => sub?.remove?.();
  }, []);

  useEffect(() => {
    logDev('bodyFront state changed', { hasUri: !!bodyFrontImage, bodyFrontImage, isPickingImage });
  }, [bodyFrontImage, isPickingImage]);

  useEffect(() => {
    if (loading) return;
    const persistDraft = async () => {
      try {
        await AsyncStorage.setItem(AVATAR_KEY, JSON.stringify({
          uri: faceImage || undefined,
          bodyFront: bodyFrontImage || undefined,
          bodySide: bodySideImage || undefined,
          avatarUrl: avatarUrl || undefined,
          rot: rotation,
          ai: aiAvatar || undefined,
          avatarStyle,
        }));
        logDev('draft autosaved', {
          hasFace: !!faceImage,
          hasBodyFront: !!bodyFrontImage,
          hasBodySide: !!bodySideImage,
        });
      } catch (e) {
        console.warn('draft autosave failed', e);
      }
    };
    persistDraft();
  }, [loading, faceImage, bodyFrontImage, bodySideImage, avatarUrl, rotation, aiAvatar, avatarStyle]);

  const applyAssetToTarget = (target, uri) => {
    if (target === 'face') {
      setFaceImage(uri);
      setAvatarUrl(null);
      setAiAvatar(null);
      return;
    }
    if (target === 'bodyFront') {
      setBodyFrontImage(uri);
      return;
    }
    if (target === 'bodySide') {
      setBodySideImage(uri);
    }
  };

  const requestPermissionOrBlock = async (kind) => {
    try {
      if (kind === 'camera') {
        const perm = await ImagePicker.requestCameraPermissionsAsync();
        if (perm?.granted) return true;
        setPermissionModal({
          visible: true,
          kind: 'camera',
          message: 'Kameratilgang kreves for å ta bilde. Åpne innstillinger og gi tilgang.',
        });
        return false;
      }
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (perm?.granted) return true;
      setPermissionModal({
        visible: true,
        kind: 'library',
        message: 'Tilgang til bilder kreves for å velge fra bibliotek. Åpne innstillinger og gi tilgang.',
      });
      return false;
    } catch (e) {
      console.warn('permission request failed', e);
      setPermissionModal({
        visible: true,
        kind,
        message: 'Kunne ikke sjekke tillatelser. Prøv igjen eller åpne innstillinger.',
      });
      return false;
    }
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const inferImageTypeFromUri = (uri) => {
    const clean = String(uri || '').split('?')[0].toLowerCase();
    if (clean.endsWith('.png')) return 'image/png';
    if (clean.endsWith('.webp')) return 'image/webp';
    if (clean.endsWith('.heic') || clean.endsWith('.heif')) return 'image/heic';
    if (clean.endsWith('.jpg') || clean.endsWith('.jpeg')) return 'image/jpeg';
    return 'image/jpeg';
  };

  const inferFileExtFromType = (type) => {
    if (type === 'image/png') return '.png';
    if (type === 'image/webp') return '.webp';
    if (type === 'image/heic') return '.heic';
    return '.jpg';
  };

  const prepareUploadAsset = async (uri, label) => {
    if (!uri) return null;
    const src = String(uri);
    const sourceInfo = await FileSystem.getInfoAsync(src, { size: true });
    if (!sourceInfo?.exists) {
      throw new Error(`${label} file does not exist: ${src}`);
    }
    const contentType = inferImageTypeFromUri(src);
    const ext = inferFileExtFromType(contentType);
    const uniqueId = `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const baseDir = FileSystem.cacheDirectory || FileSystem.documentDirectory;
    const tempPath = `${baseDir}${label}_${uniqueId}${ext}`;
    await FileSystem.copyAsync({ from: src, to: tempPath });
    const normalizedUri = `${tempPath}?upload_ts=${Date.now()}`;
    const prepared = {
      uri: normalizedUri,
      name: `${label}_${uniqueId}${ext}`,
      type: contentType,
      localPath: tempPath,
      size: sourceInfo?.size ?? null,
    };
    logDev('prepared upload asset', { label, source: src, prepared });
    return prepared;
  };

  const validatePickedAsset = async (asset) => {
    if (!asset || typeof asset.uri !== 'string' || !asset.uri.trim()) {
      return { valid: false, reason: 'missing_uri' };
    }
    const uri = asset.uri;
    // iOS/Expo can occasionally report a transient "not exists" right after picker closes.
    // Retry a couple of times before considering it invalid.
    try {
      for (let i = 0; i < 3; i += 1) {
        const info = await FileSystem.getInfoAsync(uri, { size: true });
        if (info?.exists && (typeof info.size !== 'number' || info.size > 0)) {
          return { valid: true, reason: null };
        }
        if (info?.exists && typeof info.size === 'number' && info.size <= 0) {
          if (i < 2) {
            await sleep(120);
            continue;
          }
          return { valid: false, reason: 'empty_file' };
        }
        if (i < 2) {
          await sleep(120);
          continue;
        }
      }
    } catch (e) {
      logDev('file validation error', { uri, error: String(e?.message || e) });
    }
    // Fallback: if URI is a concrete local file/document URI, allow and let upload layer verify.
    const isLikelyLocalUri = uri.startsWith('file://') || uri.startsWith('content://') || uri.startsWith('ph://');
    if (isLikelyLocalUri) return { valid: true, reason: 'fallback_uri_only' };
    return { valid: false, reason: 'file_not_found' };
  };

  const runPickerSafely = async (label, launchFn, target) => {
    const now = Date.now();
    if (now < pickerCooldownUntilRef.current) {
      logDev('picker blocked (cooldown)', { label, msLeft: pickerCooldownUntilRef.current - now });
      return;
    }
    if (pickingRef.current || isPickingImage) {
      logDev('picker blocked (already picking)', { label });
      return;
    }
    const startedAt = Date.now();
    const sessionId = ++pickerSessionRef.current;
    pickingRef.current = true;
    setIsPickingImage(true);
    setPickerNotice(null);
    logDev('picker open', { label, sessionId });
    try {
      const result = await launchFn();
      if (sessionId !== pickerSessionRef.current) {
        logDev('picker stale session ignored', { label, sessionId, current: pickerSessionRef.current });
        return;
      }
      logDev('picker result', {
        label,
        sessionId,
        canceled: !!result?.canceled,
        hasAssets: Array.isArray(result?.assets),
        assetsCount: Array.isArray(result?.assets) ? result.assets.length : 0,
        uri: result?.assets?.[0]?.uri || null,
      });
      if (result?.__permissionDenied) {
        return;
      }
      if (!result || result.canceled) {
        // Cancel is not an error: keep state and stay on same screen.
        const elapsedMs = Date.now() - startedAt;
        logDev('picker canceled', { label, elapsedMs });
        return;
      }
      const asset = Array.isArray(result.assets) ? result.assets[0] : null;
      const check = await validatePickedAsset(asset);
      if (!check.valid) {
        logDev('picker invalid asset', { label, reason: check.reason, asset });
        Alert.alert('Kunne ikke lese bildet', 'Kunne ikke lese valgt bilde. Prøv igjen.');
        return;
      }
      try {
        applyAssetToTarget(target, asset.uri);
        setPickerNotice(null);
      } catch (e) {
        console.warn('onValidAsset failed', e);
        logDev('picker apply exception', { label, error: String(e?.message || e) });
        setPickerNotice('Kunne ikke oppdatere valgt bilde.');
      }
    } catch (error) {
      console.warn(`${label} picker error`, error);
      logDev('picker exception', { label, error: String(error?.message || error) });
      Alert.alert('Bildefeil', 'Kunne ikke åpne bildebibliotek nå. Prøv igjen.');
    } finally {
      lastPickerCloseAtRef.current = Date.now();
      lastPickerLabelRef.current = label;
      pickerCooldownUntilRef.current = Date.now() + 1200;
      setTimeout(() => {
        if (sessionId === pickerSessionRef.current) {
          pickingRef.current = false;
          setIsPickingImage(false);
          logDev('picker close', { label, sessionId });
        } else {
          logDev('picker close skipped (newer session)', { label, sessionId, current: pickerSessionRef.current });
        }
      }, 250);
    }
  };

  const loadAvatar = async () => {
    try {
      const stored = await AsyncStorage.getItem(AVATAR_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.uri) setFaceImage(parsed.uri);
        if (parsed.bodyFront) setBodyFrontImage(parsed.bodyFront);
        if (parsed.bodySide) setBodySideImage(parsed.bodySide);
        if (parsed.avatarUrl) setAvatarUrl(parsed.avatarUrl);
        if (parsed.ai) setAiAvatar(parsed.ai);
        if (parsed.rot !== undefined) setRotation(parsed.rot);
        if (parsed.avatarStyle) setAvatarStyle(String(parsed.avatarStyle));
      }
    } catch (e) {
      console.warn('loadAvatar', e);
    } finally {
      setLoading(false);
    }
  };

  const analyzeFaceImage = async () => {
    if (colorAnalysisLoading) return;
    if (!faceImage) {
      Alert.alert('Mangler fjesbilde', 'Legg til et fjesbilde først.');
      return;
    }
    setColorAnalysisLoading(true);
    setColorAnalysisError(null);
    try {
      logDev('color analyze start', { AI_BACKEND_URL });
      const prepared = await prepareUploadAsset(faceImage, 'face_analysis');
      if (!prepared) throw new Error('Kunne ikke lese fjesbildet.');
      const form = new FormData();
      form.append('image', { uri: prepared.uri, name: prepared.name, type: prepared.type });
      form.append('productLimit', '12');
      const res = await fetch(`${AI_BACKEND_URL}/color/analyze-image`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data?.detail || data?.message || 'Ukjent feil';
        setColorAnalysisError(String(detail));
        setColorAnalysisLoading(false);
        return;
      }
      setColorAnalysisResult(data?.analysis || null);
    } catch (e) {
      const msg = String(e?.message || e || 'Ukjent nettverksfeil');
      setColorAnalysisError(`Klarte ikke å analysere bildet.\nBackend: ${AI_BACKEND_URL}\nFeil: ${msg}`);
    } finally {
      setColorAnalysisLoading(false);
    }
  };

  const pickFace = async () => {
    await runPickerSafely('face.gallery', async () => {
      const ok = await requestPermissionOrBlock('library');
      if (!ok) return { canceled: true, assets: [], __permissionDenied: true };
      return await ImagePicker.launchImageLibraryAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
        mediaTypes: ['images'],
        selectionLimit: 1,
      });
    }, 'face');
  };

  const takeFacePhoto = async () => {
    await runPickerSafely('face.camera', async () => {
      const ok = await requestPermissionOrBlock('camera');
      if (!ok) return { canceled: true, assets: [], __permissionDenied: true };
      return await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
      });
    }, 'face');
  };

  const pickBodyFront = async () => {
    await runPickerSafely('bodyFront.gallery', async () => {
      const ok = await requestPermissionOrBlock('library');
      if (!ok) return { canceled: true, assets: [], __permissionDenied: true };
      return await ImagePicker.launchImageLibraryAsync({
        allowsEditing: true,
        aspect: [3, 4],
        quality: 0.9,
        mediaTypes: ['images'],
        selectionLimit: 1,
      });
    }, 'bodyFront');
  };

  const takeBodyFrontPhoto = async () => {
    await runPickerSafely('bodyFront.camera', async () => {
      const ok = await requestPermissionOrBlock('camera');
      if (!ok) return { canceled: true, assets: [], __permissionDenied: true };
      return await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [3, 4],
        quality: 0.9,
      });
    }, 'bodyFront');
  };

  const pickBodySide = async () => {
    await runPickerSafely('bodySide.gallery', async () => {
      const ok = await requestPermissionOrBlock('library');
      if (!ok) return { canceled: true, assets: [], __permissionDenied: true };
      return await ImagePicker.launchImageLibraryAsync({
        allowsEditing: true,
        aspect: [3, 4],
        quality: 0.9,
        mediaTypes: ['images'],
        selectionLimit: 1,
      });
    }, 'bodySide');
  };

  const pollJob = (jobId) => {
    const start = Date.now();
    const doPoll = async () => {
      if (Date.now() - start > POLL_TIMEOUT_MS) {
        setJobError('Tidsavbrudd (3 min). Prøv igjen.');
        setGenerating(false);
        setJobStatus('failed');
        return;
      }
      try {
        const res = await fetch(`${AI_BACKEND_URL}/avatar/jobs/${jobId}`);
        const data = await res.json();
        setJobStatus(data.status);
        setJobProgress(data.progress ?? 0);
        setJobProgressMessage(data.progress_message || '');
        if (data.bodyDebug) {
          setBodyDebug(data.bodyDebug);
          if (data.bodyDebug.faceAnalysisSource && data.bodyDebug.faceAnalysisSource !== 'facemesh') {
            setPickerNotice('Kunne ikke analysere ansikt, prøv et tydeligere bilde.');
          }
        }
        if (data.error || data.error_code || data.message) {
          const parsed = resolveBackendError(data);
          setJobError(parsed.message);
          setJobErrorCode(parsed.errorCode);
          setJobRequestId(parsed.requestId);
          setJobErrorDetails(parsed.details);
        }
        if (data.status === 'done' && data.avatarUrl) {
          const url = data.avatarUrl.startsWith('http') ? data.avatarUrl : `${AI_BACKEND_URL}${data.avatarUrl}`;
          console.log('Job completed avatarUrl:', data.avatarUrl, 'resolved:', url);
          setAvatarUrl(url);
          setGenerating(false);
          setJobError(null);
          setJobErrorCode(null);
          setJobRequestId(null);
          setJobErrorDetails(null);
          return;
        }
        if (data.status === 'failed') {
          const parsed = resolveBackendError(data);
          setJobError(parsed.message);
          setJobErrorCode(parsed.errorCode);
          setJobRequestId(parsed.requestId);
          setJobErrorDetails(parsed.details);
          setAvatarUrl(null);
          Alert.alert('Generering feilet', parsed.message);
          setGenerating(false);
          return;
        }
      } catch (e) {
        console.warn('poll error', e);
        setJobError('Kunne ikke sjekke status. Prøv igjen.');
        setGenerating(false);
        return;
      }
      pollTimeoutRef.current = setTimeout(doPoll, POLL_INTERVAL_MS);
    };
    doPoll();
  };

  const generateWithAI = async () => {
    if (isPickingImage) {
      Alert.alert('Vent litt', 'Bildevelgeren er fortsatt åpen.');
      return;
    }
    if (!faceImage || !bodyFrontImage) {
      Alert.alert('Mangler bilder', 'Legg til både fjes og kropp (foran) før du genererer.');
      return;
    }
    setGenerating(true);
    setJobError(null);
    setJobErrorCode(null);
    setJobRequestId(null);
    setJobErrorDetails(null);
    setJobStatus('queued');
    setJobProgress(0);
    setJobProgressMessage('');
    setBodyDebug(null);
    try {
      const form = new FormData();
      const preparedFace = await prepareUploadAsset(faceImage, 'face');
      const preparedBodyFront = await prepareUploadAsset(bodyFrontImage, 'body_front');
      const preparedBodySide = bodySideImage ? await prepareUploadAsset(bodySideImage, 'body_side') : null;
      if (!preparedFace || !preparedBodyFront) {
        throw new Error('Mangler opplastingsdata for face/bodyFront');
      }
      form.append('face', { uri: preparedFace.uri, name: preparedFace.name, type: preparedFace.type });
      form.append('bodyFront', { uri: preparedBodyFront.uri, name: preparedBodyFront.name, type: preparedBodyFront.type });
      if (bodySideImage) {
        form.append('bodySide', { uri: preparedBodySide.uri, name: preparedBodySide.name, type: preparedBodySide.type });
      }
      form.append('avatarStyle', avatarStyle || 'neutral');
      const res = await fetch(`${AI_BACKEND_URL}/avatar/jobs`, {
        method: 'POST',
        body: form,
      });
      logDev('avatar job create headers', {
        requestId: res.headers?.get?.('x-request-id') || null,
      });
      const data = await res.json();
      if (!res.ok) {
        const parsed = resolveBackendError(data);
        setJobError(parsed.message);
        setJobErrorCode(parsed.errorCode);
        setJobRequestId(parsed.requestId);
        setJobErrorDetails(parsed.details);
        Alert.alert('Generering feilet', parsed.message);
        setGenerating(false);
        return;
      }
      const jobId = data.jobId;
      if (!jobId) {
        setJobError('Ingen jobb-ID fra server');
        setGenerating(false);
        return;
      }
      pollJob(jobId);
    } catch (e) {
      console.warn('create job error', e);
      setJobError('Kunne ikke nå server. Sjekk at backend kjører på ' + AI_BACKEND_URL);
      setGenerating(false);
    }
  };

  const chooseFaceSource = () => {
    if (isPickingImage) {
      logDev('chooseFaceSource blocked while picking');
      return;
    }
    Alert.alert('Legg til fjes', 'Velg hvordan du vil legge til fjeset:', [
      { text: 'Avbryt', style: 'cancel' },
      { text: 'Ta bilde', onPress: () => setTimeout(() => takeFacePhoto(), 150) },
      { text: 'Velg fra galleri', onPress: () => setTimeout(() => pickFace(), 150) },
    ]);
  };

  const chooseBodyFrontSource = () => {
    if (isPickingImage) {
      logDev('chooseBodyFrontSource blocked while picking');
      return;
    }
    Alert.alert('Kropp foran', 'Full kropp foran (stå rett, hel figur):', [
      { text: 'Avbryt', style: 'cancel' },
      { text: 'Ta bilde', onPress: () => setTimeout(() => takeBodyFrontPhoto(), 150) },
      { text: 'Velg fra galleri', onPress: () => setTimeout(() => pickBodyFront(), 150) },
    ]);
  };

  const openSettingsForPermission = async () => {
    try {
      await Linking.openSettings();
    } catch (e) {
      Alert.alert('Kunne ikke åpne innstillinger', 'Åpne innstillinger manuelt og gi tilgang til bilder/kamera.');
    }
  };

  const saveAvatar = async () => {
    if (!faceImage && !aiAvatar && !avatarUrl) {
      Alert.alert('Mangler avatar', 'Legg til fjes og kropp, deretter generer med AI.');
      return;
    }
    try {
      await AsyncStorage.setItem(AVATAR_KEY, JSON.stringify({
        uri: faceImage,
        bodyFront: bodyFrontImage,
        bodySide: bodySideImage,
        avatarUrl: avatarUrl || undefined,
        rot: rotation,
        ai: aiAvatar,
        avatarStyle,
      }));
      Alert.alert('Lagret', 'Avataren din er lagret.');
    } catch (e) {
      Alert.alert('Feil', 'Kunne ikke lagre.');
    }
  };

  const rot = ((rotation % 360) + 360) % 360;
  const hasAvatar = !!(avatarUrl || faceImage || bodyFrontImage);
  const canGenerate = faceImage && bodyFrontImage;
  const normalizedAvatarUrl = avatarUrl
    ? (avatarUrl.startsWith('http') ? avatarUrl : `${AI_BACKEND_URL}${avatarUrl.startsWith('/') ? '' : '/'}${avatarUrl}`)
    : null;
  const debugBaseMannequinUrl = `${AI_BACKEND_URL}/static/avatars/base_mannequin.glb`;
  const glbUrl = normalizedAvatarUrl || (jobStatus === 'failed' ? null : debugBaseMannequinUrl);
  const glbUrlWithBust = useMemo(
    () => (glbUrl ? glbUrl + (glbUrl.includes('?') ? '&' : '?') + 't=' + Date.now() : glbUrl),
    [glbUrl]
  );
  // GLBViewer preserves original materials/textures (no FORCE_SOLID).

  const colorAnalysis = useMemo(() => {
    const rgb =
      bodyDebug?.sampledSkinRGB_srgb ||
      bodyDebug?.skinColorRgb ||
      null;
    if (!Array.isArray(rgb) || rgb.length < 3) return null;
    const [r, g, b] = rgb.map((v) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return 0;
      return Math.max(0, Math.min(255, Math.round(n)));
    });
    const hex = '#' + [r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('').toUpperCase();
    return { r, g, b, hex, source: bodyDebug?.skinColorSource || bodyDebug?.faceAnalysisSource || null };
  }, [bodyDebug]);

  if (loading) return null;

  return (
    <View style={styles.screen}>
      <ScrollView contentContainerStyle={styles.container}>
      <Modal transparent visible={!!permissionModal.visible} animationType="fade">
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Tilgang kreves</Text>
            <Text style={styles.modalText}>{permissionModal.message}</Text>
            <View style={styles.modalActions}>
              <Pressable style={styles.modalBtnSecondary} onPress={() => setPermissionModal({ visible: false, kind: 'library', message: '' })}>
                <Text style={styles.modalBtnSecondaryText}>Lukk</Text>
              </Pressable>
              <Pressable style={styles.modalBtnPrimary} onPress={openSettingsForPermission}>
                <Text style={styles.modalBtnPrimaryText}>Åpne innstillinger</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      <Text style={styles.header}>Profil</Text>

      <View style={styles.segmentedWrap}>
        <Pressable
          onPress={() => setActiveTab('avatar')}
          style={[styles.segmentedItem, activeTab === 'avatar' && styles.segmentedItemActive]}
        >
          <Text style={[styles.segmentedText, activeTab === 'avatar' && styles.segmentedTextActive]}>Avatar</Text>
        </Pressable>
        <Pressable
          onPress={() => setActiveTab('color')}
          style={[styles.segmentedItem, activeTab === 'color' && styles.segmentedItemActive]}
        >
          <Text style={[styles.segmentedText, activeTab === 'color' && styles.segmentedTextActive]}>Fargeanalyse</Text>
        </Pressable>
      </View>

      {activeTab === 'avatar' ? (
        <>
          <Text style={styles.subheader}>My Avatar</Text>
          <View style={styles.avatarArea}>
            <GLBViewer
              glbUrl={glbUrlWithBust}
              rotationDeg={rot}
              onRotationChange={(deg) => setRotation(deg)}
              style={styles.glbViewer}
            />
            <Text style={styles.debugUrlText}>
              Loading GLB: {glbUrlWithBust || '(ingen URL valgt)'}
            </Text>
            <Text style={styles.debugUrlText}>
              avatarUrl state: {normalizedAvatarUrl || '(tom)'}
            </Text>
            {__DEV__ && bodyDebug?.measurementsPx && bodyDebug?.scales && (
              <Text style={styles.debugUrlText}>
                fit px: sh {bodyDebug.measurementsPx.shoulderWidthPx}, hip {bodyDebug.measurementsPx.hipWidthPx}, leg {bodyDebug.measurementsPx.legLengthPx} | scales: sh {bodyDebug.scales.shoulderWidthScale?.toFixed?.(2)}, hip {bodyDebug.scales.hipScale?.toFixed?.(2)}, leg {bodyDebug.scales.legLengthScale?.toFixed?.(2)}, torso {bodyDebug.scales.torsoLengthScale?.toFixed?.(2)}, h {bodyDebug.scales.heightScale?.toFixed?.(2)}
              </Text>
            )}
            {__DEV__ && bodyDebug?.warnings?.length > 0 && (
              <Text style={styles.debugUrlText}>
                warnings: {bodyDebug.warnings.join(', ')} | debugDeform: {String(bodyDebug.usedDebugDeform)} | bbox after: {bodyDebug.avatar_bbox_after?.x?.toFixed?.(3)} / {bodyDebug.avatar_bbox_after?.y?.toFixed?.(3)} / {bodyDebug.avatar_bbox_after?.z?.toFixed?.(3)}
              </Text>
            )}
            {__DEV__ && (bodyDebug?.skinColorRgb || bodyDebug?.sampledSkinRGB_srgb || bodyDebug?.skinMaterialsChanged) && (
              <Text style={styles.debugUrlText}>
                skin: rgb {bodyDebug.sampledSkinRGB_srgb ? bodyDebug.sampledSkinRGB_srgb.join?.(',') : bodyDebug.skinColorRgb?.join?.(',') || '(none)'} | src {bodyDebug.skinColorSource || '(?)'} | facePx {String(bodyDebug.numberOfPixelsUsed ?? bodyDebug.skinPixelsUsed ?? '(?)')} | bodyPx {String(bodyDebug.bodyPixelsUsed ?? bodyDebug.skinBodyPixelsUsed ?? '(?)')} | mats {Array.isArray(bodyDebug.skinMaterialsChanged) ? bodyDebug.skinMaterialsChanged.join(', ') : '(none)'} | inFace {bodyDebug.inputImageHash || '(?)'} | inFront {bodyDebug.inputImageHashBodyFront || '(?)'} | inSide {bodyDebug.inputImageHashBodySide || '(none)'} | outHash {bodyDebug.outputFileHash ? String(bodyDebug.outputFileHash).slice(0, 12) : '(?)'}
              </Text>
            )}
            {__DEV__ && bodyDebug?.faceAnalysisSource ? (
              <TouchableOpacity
                style={styles.debugToggleBtn}
                onPress={() => setShowFaceLikenessDebug((v) => !v)}
              >
                <Text style={styles.debugToggleBtnText}>
                  {showFaceLikenessDebug ? 'Skjul face-likeness debug' : 'Vis face-likeness debug'}
                </Text>
              </TouchableOpacity>
            ) : null}
            {__DEV__ && showFaceLikenessDebug && bodyDebug?.faceAnalysisSource ? (
              <Text style={styles.debugUrlText}>
                face likeness: {String(bodyDebug.faceLikenessEnabled)} | method {String(bodyDebug.faceLikenessMethod || 'none')} | source {String(bodyDebug.faceAnalysisSource)} | landmarks {String(bodyDebug.faceLandmarksCount ?? 0)} | morphFound {String(!!bodyDebug.hasHeadMorphTargets)} | morphCount {String(bodyDebug.morphTargetsFoundCount ?? 0)} | morphNames {(() => { try { const names = Object.values(bodyDebug.morphTargetsByMesh || bodyDebug.morphTargetNamesByMesh || {}).flat(); return names.slice(0, 8).join(', ') || '(none)'; } catch (_) { return '(none)'; } })()} | appliedMorphs {Array.isArray(bodyDebug.morphTargetsApplied) ? bodyDebug.morphTargetsApplied.slice(0, 6).map((m) => `${m.targetName || m.target}:${Number(m.weight ?? 0).toFixed?.(2) ?? m.weight}`).join(', ') : '(none)'} | overlayApplied {String(!!bodyDebug.overlayApplied)} | overlayOpacity {Number(bodyDebug.overlayOpacity ?? 0).toFixed(2)} | beardDetected {String(!!bodyDebug.beardDetected)} | beardOpacity {Number(bodyDebug.beardOpacity ?? 0).toFixed(2)} | headMeshes {Array.isArray(bodyDebug.headMeshNames) ? bodyDebug.headMeshNames.join(', ') : '(none)'} | hairMeshes {Array.isArray(bodyDebug.hairMeshNames) ? bodyDebug.hairMeshNames.join(', ') : '(none)'} | eyeMeshes {Array.isArray(bodyDebug.eyeMeshNames) ? bodyDebug.eyeMeshNames.join(', ') : '(none)'} | skinChanged {Array.isArray(bodyDebug.skinMaterialsChanged) ? bodyDebug.skinMaterialsChanged.join(', ') : '(none)'} | skinSkipped {Array.isArray(bodyDebug.materialsSkipped) ? bodyDebug.materialsSkipped.join(', ') : '(none)'} | ratios fw/h {Number(bodyDebug.faceParams?.faceWidthHeight ?? 0).toFixed(3)}, jaw {Number(bodyDebug.faceParams?.jawWidthRatio ?? 0).toFixed(3)}, cheek {Number(bodyDebug.faceParams?.cheekboneWidthRatio ?? 0).toFixed(3)}, eye {Number(bodyDebug.faceParams?.interocularRatio ?? 0).toFixed(3)}, noseW {Number(bodyDebug.faceParams?.noseWidthRatio ?? 0).toFixed(3)}, noseL {Number(bodyDebug.faceParams?.noseLengthRatio ?? 0).toFixed(3)}, mouth {Number(bodyDebug.faceParams?.mouthWidthRatio ?? 0).toFixed(3)}
              </Text>
            ) : null}
            {!normalizedAvatarUrl && (
              <TouchableOpacity style={styles.avatarOverlay} onPress={chooseFaceSource} activeOpacity={1} disabled={isPickingImage}>
                <View style={styles.avatarOverlayInner}>
                  <Ionicons name="person-add" size={40} color="#666" />
                  <Text style={styles.avatarOverlayText}>Legg til fjes og kropp</Text>
                  <Text style={styles.avatarOverlaySubtext}>Trykk for å starte</Text>
                </View>
              </TouchableOpacity>
            )}
          </View>
        </>
      ) : (
        <>
          <View style={styles.colorCard}>
            <Text style={styles.colorTitle}>Fargeanalyse</Text>
            {faceImage ? (
              <View style={styles.facePreviewRow}>
                <View style={styles.faceThumbWrap}>
                  <Image source={{ uri: faceImage }} style={styles.faceThumb} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.colorMetaText}>Bruker fjesbildet ditt</Text>
                  <Text style={styles.colorMetaSubtext}>Velg “Bytt bilde” hvis du vil bruke et annet bilde.</Text>
                </View>
              </View>
            ) : (
              <Text style={styles.colorEmptyText}>
                Last opp et fjesbilde for å få fargeanalyse + produktforslag.
              </Text>
            )}

            <TouchableOpacity
              style={[styles.aiBtn, (colorAnalysisLoading || !faceImage) && styles.aiBtnDisabled, styles.colorCta]}
              onPress={analyzeFaceImage}
              disabled={colorAnalysisLoading || !faceImage}
            >
              {colorAnalysisLoading ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.aiBtnText}>
                  {faceImage ? 'Analyser fjesbilde' : 'Legg til fjesbilde først'}
                </Text>
              )}
            </TouchableOpacity>

            <View style={styles.faceActionsRow}>
              <TouchableOpacity style={styles.smallActionBtn} onPress={chooseFaceSource} disabled={isPickingImage}>
                <Ionicons name="image-outline" size={18} color="#007AFF" />
                <Text style={styles.smallActionText}>Bytt bilde</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.smallActionBtn} onPress={() => { setColorAnalysisResult(null); setColorAnalysisError(null); }}>
                <Ionicons name="trash-outline" size={18} color="#007AFF" />
                <Text style={styles.smallActionText}>Tøm</Text>
              </TouchableOpacity>
            </View>

            {!!colorAnalysisError && (
              <View style={styles.inlineErrorRow}>
                <Text style={styles.inlineErrorText}>{colorAnalysisError}</Text>
              </View>
            )}

            {!!colorAnalysisResult && (
              <View style={styles.analysisBlock}>
                {!!faceImage && (
                  <View style={styles.analysisFaceImageWrap}>
                    <Image source={{ uri: faceImage }} style={styles.analysisFaceImage} />
                  </View>
                )}
                <Text style={styles.analysisTitle}>Resultat</Text>
                <Text style={styles.analysisLine}>Undertone: {String(colorAnalysisResult.undertone || '-')}</Text>
                <Text style={styles.analysisLine}>Kontrast: {String(colorAnalysisResult.contrast || '-')}</Text>
                <Text style={styles.analysisLine}>Sesong: {String(colorAnalysisResult.season || '-')}</Text>
                <Text style={styles.analysisLine}>Undersesong: {String(colorAnalysisResult.subseason || '-')}</Text>

                <View style={styles.paletteSection}>
                  <Text style={styles.analysisTitle}>Fargepalett</Text>
                  <ColorWheel
                    colors={[
                      ...(Array.isArray(colorAnalysisResult.best_colors)
                        ? colorAnalysisResult.best_colors.map((c) => normalizeHex(c?.hex)).filter(Boolean)
                        : []),
                      ...(Array.isArray(colorAnalysisResult.avoid_colors)
                        ? colorAnalysisResult.avoid_colors.map((c) => normalizeHex(c?.hex)).filter(Boolean)
                        : []),
                    ].slice(0, 12)}
                    size={260}
                    ringWidth={64}
                  />
                </View>

                {Array.isArray(colorAnalysisResult.best_colors) && colorAnalysisResult.best_colors.length > 0 ? (
                  <View style={[styles.colorsSection, styles.bestColorsSection]}>
                    <Text style={styles.analysisSubtitle}>Beste farger</Text>
                    <View style={styles.colorChipsRow}>
                      {colorAnalysisResult.best_colors.slice(0, 8).map((c, idx) => (
                        <View key={`best-${idx}`} style={styles.colorChip}>
                          <View style={[styles.colorDot, { backgroundColor: c?.hex || '#ccc' }]} />
                          <Text style={styles.colorChipText}>{String(c?.name || '')}</Text>
                        </View>
                      ))}
                    </View>
                  </View>
                ) : null}

                {Array.isArray(colorAnalysisResult.avoid_colors) && colorAnalysisResult.avoid_colors.length > 0 ? (
                  <View style={[styles.colorsSection, styles.avoidColorsSection]}>
                    <Text style={styles.analysisSubtitle}>Farger å unngå</Text>
                    <View style={styles.colorChipsRow}>
                      {colorAnalysisResult.avoid_colors.slice(0, 8).map((c, idx) => (
                        <View key={`avoid-${idx}`} style={styles.colorChip}>
                          <View style={[styles.colorDot, { backgroundColor: c?.hex || '#ccc' }]} />
                          <Text style={styles.colorChipText}>{String(c?.name || '')}</Text>
                        </View>
                      ))}
                    </View>
                  </View>
                ) : null}

              </View>
            )}

            {null}
          </View>

        </>
      )}

      {generating && (
        <View style={styles.progressRow}>
          <ActivityIndicator size="small" color="#5856D6" />
          <View>
            <Text style={styles.progressText}>
              {jobStatus === 'processing' || jobStatus === 'queued' ? `${jobProgress}%` : 'Starter...'}
            </Text>
            {jobProgressMessage ? (
              <Text style={styles.progressSubtext}>{jobProgressMessage}</Text>
            ) : null}
          </View>
        </View>
      )}
      {jobError && (
        <View style={styles.errorRow}>
          <Text style={styles.errorText}>{jobError}</Text>
          {__DEV__ && jobErrorCode ? <Text style={styles.debugUrlText}>error_code: {jobErrorCode}</Text> : null}
          {__DEV__ && jobRequestId ? <Text style={styles.debugUrlText}>request_id: {jobRequestId}</Text> : null}
          {__DEV__ && jobErrorDetails?.exception ? (
            <Text style={styles.debugUrlText}>exception: {String(jobErrorDetails.exception)}</Text>
          ) : null}
          {__DEV__ && jobErrorDetails?.stacktrace ? (
            <Text style={styles.debugUrlText} numberOfLines={8}>
              stacktrace: {String(jobErrorDetails.stacktrace)}
            </Text>
          ) : null}
          <TouchableOpacity style={styles.retryBtn} onPress={() => { setJobError(null); setJobErrorCode(null); setJobRequestId(null); setJobErrorDetails(null); generateWithAI(); }}>
            <Text style={styles.retryBtnText}>Prøv igjen</Text>
          </TouchableOpacity>
        </View>
      )}
      {!!pickerNotice && (
        <View style={styles.noticeRow}>
          <Text style={styles.noticeText}>{pickerNotice}</Text>
        </View>
      )}

      {activeTab === 'avatar' && hasAvatar && (
        <>
          <View style={styles.rotationRow}>
            <Text style={styles.rotationLabel}>Rotation (0–360°)</Text>
            <View style={styles.rotationControls}>
              <TouchableOpacity style={styles.arrowBtn} onPress={() => setRotation((r) => (r - 15) % 360)}>
                <Ionicons name="chevron-back" size={24} color="#333" />
              </TouchableOpacity>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={360}
                step={1}
                value={rot}
                onValueChange={(v) => setRotation(Math.round(v))}
                minimumTrackTintColor="#007AFF"
                maximumTrackTintColor="#e0e0e0"
                thumbTintColor="#007AFF"
              />
              <TouchableOpacity style={styles.arrowBtn} onPress={() => setRotation((r) => (r + 15) % 360)}>
                <Ionicons name="chevron-forward" size={24} color="#333" />
              </TouchableOpacity>
              <Text style={styles.degrees}>{rot}°</Text>
            </View>
          </View>

          <View style={styles.guidanceRow}>
            <TouchableOpacity style={[styles.guidanceBtn, isPickingImage && styles.guidanceBtnDisabled]} onPress={chooseFaceSource} disabled={isPickingImage}>
              <Ionicons name="person-outline" size={20} color="#007AFF" />
              <Text style={styles.guidanceBtnText}>Fjes</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.guidanceBtn, isPickingImage && styles.guidanceBtnDisabled]} onPress={chooseBodyFrontSource} disabled={isPickingImage}>
              <Ionicons name="body-outline" size={20} color="#007AFF" />
              <Text style={styles.guidanceBtnText}>Kropp foran</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.guidanceBtn, isPickingImage && styles.guidanceBtnDisabled]} onPress={pickBodySide} disabled={isPickingImage}>
              <Ionicons name="scan-outline" size={20} color="#007AFF" />
              <Text style={styles.guidanceBtnText}>Kropp side (valgfri)</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.styleRow}>
            <Text style={styles.styleLabel}>Avatar</Text>
            <View style={styles.styleButtons}>
              <TouchableOpacity
                style={[styles.styleBtn, avatarStyle === 'female' && styles.styleBtnActive]}
                onPress={() => setAvatarStyle('female')}
                disabled={generating}
              >
                <Text style={[styles.styleBtnText, avatarStyle === 'female' && styles.styleBtnTextActive]}>Female</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.styleBtn, avatarStyle === 'male' && styles.styleBtnActive]}
                onPress={() => setAvatarStyle('male')}
                disabled={generating}
              >
                <Text style={[styles.styleBtnText, avatarStyle === 'male' && styles.styleBtnTextActive]}>Male</Text>
              </TouchableOpacity>
            </View>
          </View>

          {hasAvatar && (
            <TouchableOpacity
              style={[styles.aiBtn, (generating || !canGenerate) && styles.aiBtnDisabled]}
              onPress={generateWithAI}
              disabled={generating}
            >
              {generating ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.aiBtnText}>
                  {canGenerate ? 'Generer 3D-avatar med AI' : 'Generer 3D-avatar (legg til Kropp foran først)'}
                </Text>
              )}
            </TouchableOpacity>
          )}

          <TouchableOpacity style={styles.changeBtn} onPress={chooseFaceSource} disabled={isPickingImage}>
            <Text style={styles.changeBtnText}>Bytt bilder</Text>
          </TouchableOpacity>
        </>
      )}

      <TouchableOpacity
        style={[styles.saveBtn, !hasAvatar && styles.saveBtnDisabled]}
        onPress={saveAvatar}
        disabled={!hasAvatar}
      >
        <Text style={styles.saveBtnText}>Lagre avatar</Text>
      </TouchableOpacity>
      </ScrollView>

      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.bottomNavItem} onPress={() => Alert.alert('Favoritter', 'Kommer snart')}>
          <Ionicons name="heart-outline" size={22} color="#111" />
          <Text style={styles.bottomNavText}>Favoritter</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.bottomNavItem} onPress={() => Alert.alert('Butikk', 'Kommer snart')}>
          <Ionicons name="bag-outline" size={22} color="#111" />
          <Text style={styles.bottomNavText}>Butikk</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.bottomNavItem, styles.bottomNavItemActive]} onPress={() => {}}>
          <Ionicons name="person" size={22} color="#111" />
          <Text style={styles.bottomNavText}>Profil</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.35)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  modalCard: {
    width: '100%',
    maxWidth: 360,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  modalTitle: {
    fontSize: 18,
    color: '#222',
    fontWeight: '700',
    marginBottom: 8,
  },
  modalText: {
    fontSize: 14,
    color: '#444',
    marginBottom: 14,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 10,
  },
  modalActionsCol: {
    gap: 10,
  },
  modalBtnPrimary: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalBtnPrimaryText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  modalBtnSecondary: {
    backgroundColor: '#f0f0f0',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalBtnSecondaryText: {
    color: '#333',
    fontSize: 14,
    fontWeight: '600',
  },
  screen: {
    flex: 1,
    backgroundColor: '#fff',
  },
  container: {
    flexGrow: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    paddingTop: 50,
    paddingBottom: 120,
  },
  bottomNav: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingBottom: 24,
    paddingTop: 10,
    paddingHorizontal: 16,
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.08)',
  },
  bottomNavItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 6,
  },
  bottomNavItemActive: {
    opacity: 1,
  },
  bottomNavText: {
    marginTop: 4,
    fontSize: 10,
    color: '#111',
    fontWeight: '600',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subheader: {
    fontSize: 16,
    color: '#666',
    marginBottom: 24,
    fontWeight: '600',
  },
  segmentedWrap: {
    width: '92%',
    maxWidth: 360,
    flexDirection: 'row',
    backgroundColor: '#efeff4',
    borderRadius: 16,
    padding: 4,
    marginTop: 12,
    marginBottom: 18,
  },
  segmentedItem: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  segmentedItemActive: {
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  segmentedText: {
    fontSize: 14,
    color: '#6b6b6b',
    fontWeight: '600',
  },
  segmentedTextActive: {
    color: '#111',
  },
  avatarArea: {
    width: '100%',
    minHeight: 360,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 24,
    position: 'relative',
  },
  colorCard: {
    width: '92%',
    maxWidth: 420,
    backgroundColor: '#f7f7fb',
    borderRadius: 16,
    padding: 16,
    marginTop: 56,
    marginBottom: 14,
  },
  colorTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#222',
    marginBottom: 12,
  },
  colorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  colorSwatch: {
    width: 44,
    height: 44,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  colorMeta: {
    flex: 1,
  },
  colorMetaText: {
    fontSize: 14,
    color: '#222',
    fontWeight: '600',
  },
  colorMetaSubtext: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  colorEmptyText: {
    fontSize: 14,
    color: '#444',
    lineHeight: 20,
    marginBottom: 12,
  },
  colorCta: {
    marginTop: 2,
  },
  facePreviewRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  faceThumbWrap: {
    width: 52,
    height: 52,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#eee',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  faceThumb: {
    width: '100%',
    height: '100%',
  },
  faceActionsRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 6,
    marginBottom: 6,
  },
  smallActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#fff',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  smallActionText: {
    fontSize: 13,
    color: '#007AFF',
    fontWeight: '600',
  },
  inlineErrorRow: {
    marginTop: 8,
    padding: 10,
    backgroundColor: '#ffebee',
    borderRadius: 10,
  },
  inlineErrorText: {
    fontSize: 13,
    color: '#c62828',
    fontWeight: '600',
  },
  analysisBlock: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.06)',
  },
  analysisFaceImageWrap: {
    width: '100%',
    borderRadius: 58,
    overflow: 'hidden',
    backgroundColor: 'transparent',
    marginBottom: 18,
  },
  analysisFaceImage: {
    width: '100%',
    height: 260,
    resizeMode: 'contain',
    borderRadius: 58,
    backgroundColor: 'transparent',
  },
  analysisTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#222',
    marginBottom: 8,
  },
  analysisSubtitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#222',
    marginBottom: 8,
  },
  analysisLine: {
    fontSize: 13,
    color: '#333',
    marginBottom: 4,
    fontWeight: '600',
  },
  paletteSection: {
    marginTop: 40,
    alignItems: 'center',
  },
  paletteWheelWrap: {
    marginTop: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  colorsSection: {
    marginTop: 10,
  },
  bestColorsSection: {
    marginTop: 40,
  },
  avoidColorsSection: {
    marginTop: 30,
  },
  colorChipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  colorChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    backgroundColor: '#f7f7fb',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.06)',
  },
  colorDot: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.12)',
  },
  colorChipText: {
    fontSize: 12,
    color: '#222',
    fontWeight: '700',
  },
  avatarOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255,255,255,0.85)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarOverlayInner: {
    alignItems: 'center',
    padding: 24,
  },
  avatarOverlayText: {
    fontSize: 16,
    color: '#333',
    marginTop: 12,
    fontWeight: '600',
  },
  avatarOverlaySubtext: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  glbViewer: {
    width: 280,
    height: 360,
  },
  debugUrlText: {
    marginTop: 10,
    marginHorizontal: 16,
    fontSize: 12,
    color: '#222',
    textAlign: 'center',
  },
  debugToggleBtn: {
    marginTop: 8,
    paddingVertical: 6,
    paddingHorizontal: 10,
    backgroundColor: '#e8f2ff',
    borderRadius: 8,
  },
  debugToggleBtnText: {
    fontSize: 12,
    color: '#1e4f8a',
    fontWeight: '600',
  },
  avatarRotateWrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  progressRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 12,
  },
  progressSubtext: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  progressText: {
    fontSize: 14,
    color: '#666',
  },
  errorRow: {
    marginHorizontal: 20,
    marginBottom: 12,
    padding: 12,
    backgroundColor: '#ffebee',
    borderRadius: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#c62828',
    marginBottom: 8,
  },
  noticeRow: {
    marginHorizontal: 20,
    marginBottom: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#e8f2ff',
    borderRadius: 8,
  },
  noticeText: {
    fontSize: 13,
    color: '#1e4f8a',
  },
  retryBtn: {
    alignSelf: 'flex-start',
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: '#c62828',
    borderRadius: 8,
  },
  retryBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  guidanceRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 16,
  },
  styleRow: {
    width: '90%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  styleLabel: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
  },
  styleButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  styleBtn: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#d0d0d0',
    backgroundColor: '#fff',
  },
  styleBtnActive: {
    borderColor: '#007AFF',
    backgroundColor: 'rgba(0,122,255,0.08)',
  },
  styleBtnText: {
    fontSize: 13,
    color: '#333',
    fontWeight: '600',
  },
  styleBtnTextActive: {
    color: '#007AFF',
  },
  guidanceBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#f0f0f0',
    borderRadius: 20,
  },
  guidanceBtnDisabled: {
    opacity: 0.55,
  },
  guidanceBtnText: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '500',
  },
  slider: {
    flex: 1,
    height: 40,
  },
  avatarComposite: {
    alignItems: 'center',
  },
  faceCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    overflow: 'hidden',
    borderWidth: 4,
    borderColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 8,
  },
  faceImage: {
    width: '100%',
    height: '100%',
  },
  bodyWrapper: {
    marginTop: -20,
    width: 100,
    position: 'relative',
  },
  avatarBody: {
    width: 100,
    height: 160,
    backgroundColor: '#e8e4e0',
    borderTopLeftRadius: 50,
    borderTopRightRadius: 50,
  },
  aiAvatar: {
    width: 280,
    height: 400,
  },
  aiBtn: {
    backgroundColor: '#5856D6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 24,
    marginBottom: 16,
  },
  aiBtnDisabled: {
    opacity: 0.7,
  },
  aiBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  placeholder: {
    width: 220,
    height: 280,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#ddd',
  },
  placeholderText: {
    fontSize: 16,
    color: '#333',
    marginTop: 12,
    fontWeight: '500',
  },
  placeholderSubtext: {
    fontSize: 14,
    color: '#888',
    marginTop: 4,
  },
  rotationRow: {
    width: '92%',
    maxWidth: 420,
    marginTop: 8,
    marginBottom: 24,
  },
  rotationLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  rotationControls: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  arrowBtn: {
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
  },
  sliderTrack: {
    flex: 1,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  sliderFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  degrees: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    minWidth: 44,
    textAlign: 'right',
  },
  changeBtn: {
    marginBottom: 24,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  changeBtnText: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
  saveBtn: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 48,
    borderRadius: 30,
  },
  saveBtnDisabled: {
    opacity: 0.5,
  },
  saveBtnText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
