export default function ContextPanel({ response }) {
  const context = response?.context;
  if (!context) return null;

  return (
    <section className="context-panel">
      <div>
        <span className="meta-label">Rasa</span>
        <strong>{response.detected_rasa || 'Not required'}</strong>
      </div>
      <div>
        <span className="meta-label">Bhava</span>
        <strong>{response.detected_bhava || 'Curated mode'}</strong>
      </div>
      <div>
        <span className="meta-label">Prahara</span>
        <strong>{context.prahara} ({context.arc})</strong>
      </div>
      <div>
        <span className="meta-label">Ritu</span>
        <strong>{context.ritu || 'unknown'}</strong>
      </div>
      <div>
        <span className="meta-label">Weather</span>
        <strong>{context.weather_condition || 'unknown'}</strong>
      </div>
    </section>
  );
}

