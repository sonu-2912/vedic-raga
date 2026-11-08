import { Activity, BookOpen, Flame, Moon, Sparkles, Waves } from 'lucide-react';

const ICONS = {
  therapeutic: Waves,
  traditional: Sparkles,
  study: BookOpen,
  gym: Flame,
  sleep: Moon,
  meditation: Activity,
};

export default function ModeSelector({ modes, selectedMode, onSelect }) {
  return (
    <div className="mode-grid">
      {modes.map((mode) => {
        const Icon = ICONS[mode.id] || Sparkles;
        const selected = selectedMode === mode.id;
        return (
          <button
            className={`mode-card ${selected ? 'selected' : ''}`}
            key={mode.id}
            type="button"
            onClick={() => onSelect(mode.id)}
          >
            <span className="mode-icon"><Icon size={22} /></span>
            <span className="mode-name">{mode.name}</span>
            <span className="mode-description">{mode.description}</span>
          </button>
        );
      })}
    </div>
  );
}

