import { MapPin, Send } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getModes, recommendActivity, recommendFromImage } from '../api/recommendations.js';
import CameraCapture from '../components/CameraCapture.jsx';
import ModeSelector from '../components/ModeSelector.jsx';

const DEFAULT_LOCATION = {
  lat: 28.6139,
  lon: 77.209,
  tz: 'Asia/Kolkata',
};

export default function Capture() {
  const navigate = useNavigate();
  const [modes, setModes] = useState([]);
  const [mode, setMode] = useState(sessionStorage.getItem('selected_mode') || 'therapeutic');
  const [style, setStyle] = useState('both');
  const [image, setImage] = useState(null);
  const [location, setLocation] = useState(DEFAULT_LOCATION);
  const [locationStatus, setLocationStatus] = useState('Default location');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const selectedMode = useMemo(() => modes.find((item) => item.id === mode), [modes, mode]);
  const requiresImage = selectedMode?.requires_image ?? true;

  useEffect(() => {
    getModes().then((response) => setModes(response.modes)).catch((err) => setError(err.message));
    resolveLocation();
  }, []);

  async function resolveLocation() {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_LOCATION.tz;
    if (!navigator.geolocation) {
      setLocation({ ...DEFAULT_LOCATION, tz });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          tz,
        });
        setLocationStatus('Location ready');
      },
      () => {
        setLocation({ ...DEFAULT_LOCATION, tz });
        setLocationStatus('Using default location');
      },
      { enableHighAccuracy: false, timeout: 6000, maximumAge: 300000 },
    );
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    if (requiresImage && !image) {
      setError('Capture or upload an image for this mode.');
      return;
    }

    setLoading(true);
    try {
      const response = requiresImage
        ? await recommendFromImage({ image, mode, style, location })
        : await recommendActivity({ mode, style, location });
      sessionStorage.setItem('latest_recommendation', JSON.stringify(response));
      navigate('/recommendations', { state: { response } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="stack">
      <div className="page-heading">
        <span className="eyebrow">Samaya and bhava</span>
        <h1>Prepare recommendation</h1>
      </div>

      <form className="capture-layout" onSubmit={handleSubmit}>
        <div className="panel-block">
          <ModeSelector modes={modes} selectedMode={mode} onSelect={setMode} />
        </div>

        {requiresImage ? <CameraCapture onImageReady={setImage} /> : null}

        <div className="settings-panel">
          <div>
            <span className="field-label">Style</span>
            <div className="segmented">
              {['both', 'vocal', 'instrumental'].map((item) => (
                <button
                  className={style === item ? 'active' : ''}
                  key={item}
                  type="button"
                  onClick={() => setStyle(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="location-line">
            <MapPin size={18} />
            <span>{locationStatus}</span>
          </div>

          {error ? <p className="form-error">{error}</p> : null}

          <button className="primary-button full-width" type="submit" disabled={loading}>
            <Send size={18} />
            {loading ? 'Recommending' : 'Recommend ragas'}
          </button>
        </div>
      </form>
    </section>
  );
}

