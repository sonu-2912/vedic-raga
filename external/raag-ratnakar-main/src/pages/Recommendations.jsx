import { ArrowLeft, AlertTriangle } from 'lucide-react';
import { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ContextPanel from '../components/ContextPanel.jsx';
import RagaCard from '../components/RagaCard.jsx';

export default function Recommendations() {
  const navigate = useNavigate();
  const location = useLocation();
  const response = useMemo(() => {
    if (location.state?.response) return location.state.response;
    const raw = sessionStorage.getItem('latest_recommendation');
    return raw ? JSON.parse(raw) : null;
  }, [location.state]);

  if (!response) {
    return (
      <section className="empty-state">
        <h1>No recommendation found</h1>
        <button className="primary-button" type="button" onClick={() => navigate('/dashboard')}>
          <ArrowLeft size={18} />
          Dashboard
        </button>
      </section>
    );
  }

  return (
    <section className="stack">
      <div className="page-heading row-heading">
        <div>
          <span className="eyebrow">{response.mode}</span>
          <h1>Recommended Ragas</h1>
        </div>
        <button className="secondary-button" type="button" onClick={() => navigate('/capture')}>
          <ArrowLeft size={18} />
          Back
        </button>
      </div>

      {response.low_confidence ? (
        <div className="warning-band">
          <AlertTriangle size={18} />
          <span>Detected rasa is uncertain. The list uses the closest bhava match.</span>
        </div>
      ) : null}

      <ContextPanel response={response} />

      <div className="result-note">
        Primary bhava matches are ranked above secondary matches. Therapeutic mode shows the top 2 ragas with a small song set, while traditional mode keeps a small raga list with 2 songs per selection.
      </div>

      <div className="raga-list">
        {response.recommendations.map((recommendation, index) => (
          <RagaCard recommendation={recommendation} index={index} key={recommendation.raga_name} />
        ))}
      </div>
    </section>
  );
}
