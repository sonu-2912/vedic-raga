const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function getToken() {
  return localStorage.getItem('raga_token');
}

export function setSession(authResponse) {
  localStorage.setItem('raga_token', authResponse.access_token);
  localStorage.setItem('raga_user', JSON.stringify(authResponse.user));
}

export function clearSession() {
  localStorage.removeItem('raga_token');
  localStorage.removeItem('raga_user');
  sessionStorage.removeItem('latest_recommendation');
}

export function getUser() {
  const raw = localStorage.getItem('raga_user');
  return raw ? JSON.parse(raw) : null;
}

export async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();

  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  const contentType = response.headers.get('content-type') || '';
  const body = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof body === 'object' && body.detail ? body.detail : 'Request failed';
    throw new Error(Array.isArray(message) ? message.map((item) => item.msg).join(', ') : message);
  }
  return body;
}

