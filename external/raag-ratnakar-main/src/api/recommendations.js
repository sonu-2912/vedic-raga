import { apiRequest } from './client.js';

export async function getModes() {
  return apiRequest('/api/modes');
}

export async function recommendFromImage({ image, mode, style, location }) {
  const form = new FormData();
  form.append('image', image, 'capture.jpg');
  form.append('mode', mode);
  form.append('style', style);
  form.append('lat', String(location.lat));
  form.append('lon', String(location.lon));
  form.append('tz', location.tz);

  return apiRequest('/api/recommend/image', {
    method: 'POST',
    body: form,
  });
}

export async function recommendActivity({ mode, style, location }) {
  return apiRequest('/api/recommend/activity', {
    method: 'POST',
    body: JSON.stringify({
      mode,
      style,
      lat: location.lat,
      lon: location.lon,
      tz: location.tz,
    }),
  });
}

export async function getHistory() {
  return apiRequest('/api/recommend/history');
}

