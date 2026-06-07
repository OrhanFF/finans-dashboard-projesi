import React from "react";

function TweetCard({ tweet }) {
  const isToxic = tweet.is_toxic;

  // Spam ise hafif şeffaflık verelim (filtre kapalıyken ayırt edilsin)
  const cardStyle = isToxic ? { opacity: 0.8, border: '1px solid #f4212e' } : {};

  return (
    <article className={`tweet-card ${isToxic ? 'toxic-warning' : ''}`} style={cardStyle}>
      <header className='tweet-card-header'>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <strong className='tweet-user'>@{tweet.user}</strong>
          {isToxic && (
            <span className="toxic-badge">⚠️ Olası Spam/Toksik</span>
          )}
        </div>
        <time dateTime={tweet.timestamp} className='tweet-date'>
          {new Date(tweet.timestamp).toLocaleString('tr-TR')}
        </time>
      </header>
      <p className='tweet-content'>
        {tweet.tweet}
      </p>
      {isToxic && (
        <div className="toxicity-score">
          Yapay Zeka Güven Puanı: %{((1 - tweet.toxicity_score) * 100).toFixed(0)}
        </div>
      )}
    </article>
  );
}

export default TweetCard;