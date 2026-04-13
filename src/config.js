// AI-backend URL. For fysisk telefon: sett til PCens LAN-IP, f.eks. 'http://192.168.1.100:8000'
const DEV_BACKEND_HOST = 'http://10.0.0.27:8000'; // PCens IPv4 (ipconfig) – endres ved annet nett / hotspot
const AI_BACKEND_URL = __DEV__
  ? DEV_BACKEND_HOST
  : 'https://din-ai-backend.up.railway.app';

export { AI_BACKEND_URL };
