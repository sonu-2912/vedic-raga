function toEmbedUrl(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes('youtube.com')) {
      const id = parsed.searchParams.get('v');
      return id ? `https://www.youtube.com/embed/${id}` : '';
    }
    if (parsed.hostname.includes('youtu.be')) {
      const id = parsed.pathname.replace('/', '');
      return id ? `https://www.youtube.com/embed/${id}` : '';
    }
  } catch {
    return '';
  }
  return '';
}

export default function RagaCard({ recommendation, index }) {
  const allLinks = [
    ...(recommendation.youtube_links?.vocal || []),
    ...(recommendation.youtube_links?.instrumental || []),
  ];

  return (
    <article className="raga-card">
      <div className="raga-card-main">
        <div className="raga-rank">{index + 1}</div>
        <div>
          <h3>{recommendation.raga_name}</h3>
          <div className="raga-tags">
            <span>{recommendation.bhava_match_type}</span>
            <span>score {recommendation.score.toFixed(3)}</span>
            <span>target {recommendation.target_bhava}</span>
          </div>
        </div>
      </div>

      <div className="score-line">
        <span>Bhava {recommendation.bhava_score}</span>
        <span>Prahara {recommendation.prahara_score}</span>
        <span>Ritu {recommendation.ritu_score}</span>
      </div>

      <div className="video-column">
        {allLinks.map((link, idx) => {
          const embedUrl = toEmbedUrl(link);
          if (!embedUrl) return null;
          return (
            <div className="video-frame" key={`${link}-${idx}`}>
              <iframe
                src={embedUrl}
                title={`${recommendation.raga_name} performance ${idx + 1}`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          );
        })}
      </div>
    </article>
  );
}

