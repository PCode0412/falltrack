import React from 'react';
import { BarChart3, Activity, Layers, Database, BookOpen } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'benchmark', label: 'Model Benchmark', icon: BarChart3 },
    { id: 'sensor', label: 'Sensor Stream & Alert', icon: Activity },
    { id: 'sequence', label: 'Sequence Timeline', icon: Layers },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'methodology', label: 'Methodology & Math', icon: BookOpen }
  ];

  return (
    <header className="header">
      <div className="header-container">
        <div className="brand-section">
          <div className="brand-logo">
            <Activity size={24} />
          </div>
          <div>
            <div className="brand-title">SisFall HealthAI</div>
            <div className="brand-subtitle">Wearable Fall Detection & Hidden Markov Model Benchmark</div>
          </div>
        </div>

        <nav className="nav-tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`nav-tab-button ${isActive ? 'active' : ''}`}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
