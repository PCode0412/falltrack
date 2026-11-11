import React, { useState, useEffect, Component } from 'react';
import Navbar from './components/Navbar';
import ModelComparison from './components/ModelComparison';
import SensorStream from './components/SensorStream';
import SequenceTimeline from './components/SequenceTimeline';
import DatasetExplorer from './components/DatasetExplorer';
import Methodology from './components/Methodology';
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import './App.css';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ borderLeft: '4px solid #DC2626', background: '#FEF2F2', margin: '2rem 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#991B1B' }}>
            <AlertCircle size={24} />
            <div>
              <div style={{ fontWeight: 700 }}>Component Rendering Notice</div>
              <div style={{ fontSize: '0.875rem' }}>{this.state.error?.message || 'An unexpected rendering error occurred.'}</div>
            </div>
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="badge badge-primary"
            style={{ marginTop: '1rem', cursor: 'pointer', padding: '0.4rem 0.8rem' }}
          >
            Retry View
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('benchmark');
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [benchRes, recRes, subRes, actRes] = await Promise.all([
          fetch('/data/benchmark.json'),
          fetch('/data/recordings.json'),
          fetch('/data/subjects.json'),
          fetch('/data/activities.json')
        ]);

        if (!benchRes.ok || !recRes.ok || !subRes.ok || !actRes.ok) {
          throw new Error('Failed to load dataset files. Please verify data in /public/data.');
        }

        const benchJson = await benchRes.json();
        const recJson = await recRes.json();
        const subJson = await subRes.json();
        const actJson = await actRes.json();

        setBenchmarkData(benchJson);
        setRecordings(recJson);
        setSubjects(subJson);
        setActivities(actJson);
        setLoading(false);
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div className="app-wrapper">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="main-content">
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '1rem' }}>
            <Loader2 size={36} className="animate-spin" color="#2563EB" />
            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#475569' }}>
              Loading SisFall Benchmark Dataset & Models...
            </div>
          </div>
        )}

        {error && (
          <div className="card" style={{ borderLeft: '4px solid #DC2626', background: '#FEF2F2' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#991B1B' }}>
              <AlertCircle size={24} />
              <div>
                <div style={{ fontWeight: 700 }}>Initialization Error</div>
                <div style={{ fontSize: '0.875rem' }}>{error}</div>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && (
          <ErrorBoundary key={activeTab}>
            {activeTab === 'benchmark' && <ModelComparison benchmarkData={benchmarkData} />}
            {activeTab === 'sensor' && <SensorStream recordings={recordings} />}
            {activeTab === 'sequence' && <SequenceTimeline recordings={recordings} />}
            {activeTab === 'dataset' && <DatasetExplorer subjects={subjects} activities={activities} />}
            {activeTab === 'methodology' && <Methodology />}
          </ErrorBoundary>
        )}
      </main>

      <footer className="footer">
        <div>
          <strong>SisFall Fall Detection & Activity Recognition Benchmark</strong> • Subject-Independent Evaluation Framework
        </div>
        <div style={{ marginTop: '0.25rem', color: '#94A3B8' }}>
          Wearable Biomedical Computing Lab • Gaussian Hidden Markov Model with Log-Space Viterbi Path Optimization
        </div>
      </footer>
    </div>
  );
}
