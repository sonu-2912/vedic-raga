import { apiRequest, setSession } from './client.js';

export async function login(email, password) {
  const response = await apiRequest('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setSession(response);
  return response;
}

export async function register(email, password) {
  const response = await apiRequest('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setSession(response);
  return response;
}

