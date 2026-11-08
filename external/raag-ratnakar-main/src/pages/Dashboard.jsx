import { ArrowRight, Clock3 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory, getModes } from '../api/recommendations.js';
import ModeSelector from '../components/ModeSelector.jsx';

export default function Dashboard() {
  const navigate = useNavigate();
  const [modes, setModes] = useState([]);
  const [selectedMode, setSelectedMode] = useState('therapeutic');
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    async function load() {
      try {
        const [modeResponse, historyResponse] = await Promise.all([getModes(), getHistory()]);
        setModes(modeResponse.modes);
        setHistory(historyResponse.items || []);
        const defaultMode = modeResponse.modes.find((mode) => mode.default)?.id;
        setSelectedMode(defaultMode || modeResponse.modes[0]?.id || 'therapeutic');
      } catch (err) {
        setError(err.message);
      }
    }
    load();
  }, []);

  function continueToCapture() {
    sessionStorage.setItem('selected_mode', selectedMode);
    navigate('/capture');
  }

  return (
    <section className="stack">
      <div className="page-heading">
        <span className="eyebrow">Nava rasa</span>
        <h1>Choose your listening path</h1>
      </div>

      {error ? <p className="form-error">{error}</p> : null}

      <ModeSelector modes={modes} selectedMode={selectedMode} onSelect={setSelectedMode} />

      <div className="action-row">
        <button className="primary-button" type="button" onClick={continueToCapture}>
          <ArrowRight size={18} />
          Continue
        </button>
      </div>

      <section className="history-section">
        <div className="section-title">
          <Clock3 size={18} />
          <h2>Recent sessions</h2>
        </div>
        {history.length ? (
          <div className="history-list">
            {history.slice(0, 5).map((item) => (
              <div className="history-item" key={item.id}>
                <strong>{item.mode}</strong>
                <span>{item.detected_bhava || 'curated'} · prahara {item.prahara || '-'}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No sessions yet.</p>
        )}
      </section>
    </section>
  );
}

