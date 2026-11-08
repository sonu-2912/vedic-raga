import { LogOut } from 'lucide-react';
import { Outlet, useNavigate } from 'react-router-dom';
import { clearSession, getUser } from '../api/client.js';

export default function AppShell() {
  const navigate = useNavigate();
  const user = getUser();

  function handleLogout() {
    clearSession();
    navigate('/login');
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-button" type="button" onClick={() => navigate('/dashboard')}>
          <span className="brand-mark">
            <img src="/icon.png" alt="" />
          </span>
          <span>
            <span className="brand-title">Raag Ratnakar</span>
            <span className="brand-subtitle">Where Every Mood Finds a Raga</span>
          </span>
        </button>
        <div className="topbar-actions">
          <span className="user-email">{user?.email}</span>
          <button className="icon-button" type="button" onClick={handleLogout} title="Logout">
            <LogOut size={18} />
          </button>
        </div>
      </header>
      <main className="page-frame">
        <Outlet />
      </main>
    </div>
  );
}
