import { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import TweetCard from './TweetCard';
import Button from './Button';
import StockInfoPage from './StockInfoPage';
import NewsPage from './NewsPage';
import AnalystPage from './AnalystPage';
import PredictionPage from './PredictionPage';

const API_URL = "http://localhost:3001/api";

function App() {
  const [searchTerm, setSearchTerm]   = useState("");
  const [results, setResults]         = useState([]);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [activeView, setActiveView]   = useState('tweets');
  const [isSearched, setIsSearched]   = useState(false);

  const [localSearch, setLocalSearch] = useState("");
  const [userFilter, setUserFilter]   = useState("");
  const [sortOrder, setSortOrder]     = useState("newest");
  const [hideToxic, setHideToxic]     = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const tweetsPerPage = 10;

  useEffect(() => {
    setCurrentPage(1);
  }, [localSearch, userFilter, sortOrder, hideToxic]);

  const handleSearch = async (e) => {
    e.preventDefault();
    const cleanSearchTerm = searchTerm.trim().replace(/^#/, '');
    if (cleanSearchTerm === "") return;

    setLoading(true);
    setResults([]);
    setError(null);
    setIsSearched(false);
    setActiveView('tweets');

    try {
      const response = await axios.get(`${API_URL}/search`, {
        params: { q: cleanSearchTerm }
      });
      setResults(response.data);
    } catch (err) {
      setError("Veriler çekilirken bir hata oluştu");
      console.error(err);
    } finally {
      setLoading(false);
      setIsSearched(true);
    }
  };

  const filteredResults = useMemo(() => {
    return results
      .filter(tweet => {
        const matchesContent = tweet.tweet.toLowerCase().includes(localSearch.toLowerCase());
        const matchesUser    = tweet.user.toLowerCase().includes(userFilter.toLowerCase());
        const matchesToxic   = hideToxic ? !tweet.is_toxic : true;
        return matchesContent && matchesUser && matchesToxic;
      })
      .sort((a, b) => {
        const dateA = new Date(a.timestamp);
        const dateB = new Date(b.timestamp);
        return sortOrder === "newest" ? dateB - dateA : dateA - dateB;
      });
  }, [results, localSearch, userFilter, sortOrder, hideToxic]);

  const toxicCount = useMemo(() => {
    return results.filter(t => t.is_toxic).length;
  }, [results]);

  const indexOfLastTweet  = currentPage * tweetsPerPage;
  const indexOfFirstTweet = indexOfLastTweet - tweetsPerPage;
  const currentTweets     = filteredResults.slice(indexOfFirstTweet, indexOfLastTweet);
  const totalPages        = Math.ceil(filteredResults.length / tweetsPerPage);

  return (
    <div className='app-container'>
      <header className='app-header'>
        <h1 className='app-title'>Gündem Borsa AI</h1>
        <form className='search-form' onSubmit={handleSearch}>
          <input
            className='search-input'
            type="text"
            placeholder="Hisse sembolü ara (örn: AAPL)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            disabled={loading}
          />
          <button
            className='search-button'
            type="submit"
            disabled={loading || searchTerm.trim() === ""}
          >
            {loading ? 'Aranıyor...' : 'Ara'}
          </button>
        </form>
      </header>

      <main className='content-container'>
        {error   && <div className='error-message'><p>{error}</p></div>}
        {loading && <div className='loading-indicator'><p>Sonuçlar yükleniyor...</p></div>}

        {!loading && !error && isSearched && (
          <>
            <div className="options-bar">
              <Button name="X Yorumları"       onClick={() => setActiveView('tweets')}     isActive={activeView === 'tweets'} />
              <Button name="Hisse Bilgileri"   onClick={() => setActiveView('stocks')}     isActive={activeView === 'stocks'} />
              <Button name="Haberler"          onClick={() => setActiveView('news')}       isActive={activeView === 'news'} />
              <Button name="Analist"           onClick={() => setActiveView('analyst')}    isActive={activeView === 'analyst'} />
              <Button name="AI Tahmin"         onClick={() => setActiveView('prediction')} isActive={activeView === 'prediction'} />
            </div>

            {activeView === 'tweets' && (
              <>
                <div className="tweet-filter-panel tweet-card">
                  <div className="filter-group">
                    <input
                      type="text"
                      placeholder="Tweet içinde ara..."
                      className="mini-input"
                      value={localSearch}
                      onChange={(e) => setLocalSearch(e.target.value)}
                    />
                    <input
                      type="text"
                      placeholder="@kullanıcı ara..."
                      className="mini-input"
                      value={userFilter}
                      onChange={(e) => setUserFilter(e.target.value)}
                    />
                  </div>
                  <div className="filter-group">
                    <select
                      value={sortOrder}
                      onChange={(e) => setSortOrder(e.target.value)}
                      className="mini-select"
                    >
                      <option value="newest">En Yeni</option>
                      <option value="oldest">En Eski</option>
                    </select>

                    <button
                      className={`mini-btn ${hideToxic ? 'active' : ''}`}
                      onClick={() => setHideToxic(prev => !prev)}
                      title="Toksik/spam tweetleri gizle"
                    >
                      {hideToxic ? '🛡️ Filtre Açık' : '⚠️ Filtre Kapalı'}
                      {toxicCount > 0 && (
                        <span className="toxic-count-badge">{toxicCount}</span>
                      )}
                    </button>
                  </div>

                  <div className="filter-summary">
                    Toplam <strong>{filteredResults.length}</strong> tweet bulundu.
                    {toxicCount > 0 && (
                      <span className="toxic-summary-text">
                        ({toxicCount} toksik {hideToxic ? '— gizlendi' : '— gösteriliyor'})
                      </span>
                    )}
                    {' '}(Sayfa {currentPage} / {totalPages || 1})
                  </div>
                </div>

                {filteredResults.length > 0 ? (
                  <>
                    <section className='results-list'>
                      {currentTweets.map((tweet) => (
                        <TweetCard key={tweet.id} tweet={tweet} />
                      ))}
                    </section>
                    <div className="pagination">
                      <button onClick={() => setCurrentPage(p => p - 1)} disabled={currentPage === 1} className="page-btn">
                        &laquo; Önceki
                      </button>
                      <span className="page-info">{currentPage}</span>
                      <button onClick={() => setCurrentPage(p => p + 1)} disabled={currentPage === totalPages} className="page-btn">
                        Sonraki &raquo;
                      </button>
                    </div>
                  </>
                ) : (
                  <div className='no-results'><p>Filtrelere uygun tweet bulunamadı.</p></div>
                )}
              </>
            )}

            {activeView === 'stocks'     && <StockInfoPage  searchTerm={searchTerm} />}
            {activeView === 'news'       && <NewsPage       searchTerm={searchTerm} />}
            {activeView === 'analyst'    && <AnalystPage    searchTerm={searchTerm} />}
            {activeView === 'prediction' && <PredictionPage searchTerm={searchTerm} />}
          </>
        )}
      </main>
    </div>
  );
}

export default App;