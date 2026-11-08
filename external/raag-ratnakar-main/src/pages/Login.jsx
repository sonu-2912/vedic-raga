import { LogIn } from 'lucide-react';
import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { login } from '../api/auth.js';

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-heading">
          <span className="eyebrow">Swarasthana</span>
          <h1>Raga Recommender</h1>
        </div>
        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Email
            <input value={email} type="email" onChange={(event) => setEmail(event.target.value)} required />
          </label>
          <label>
            Password
            <input
              value={password}
              type="password"
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={6}
            />
          </label>
          {error ? <p className="form-error">{error}</p> : null}
          <button className="primary-button full-width" type="submit" disabled={loading}>
            <LogIn size={18} />
            {loading ? 'Signing in' : 'Sign in'}
          </button>
        </form>
        <p className="auth-switch">
          New here? <Link to="/register">Create account</Link>
        </p>
      </section>
    </main>
  );
}

