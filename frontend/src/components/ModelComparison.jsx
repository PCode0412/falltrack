import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Award, CheckCircle, Zap, TrendingUp, ShieldCheck } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function ModelComparison({ benchmarkData }) {
  const [metricView, setMetricView] = useState('event'); // 'event' or 'window'

  if (!benchmarkData || !benchmarkData.models) {
    return <div className="card">Loading benchmark data...</div>;
  }

  const models = benchmarkData.models;
  const modelNames = ['Gaussian HMM', 'Random Forest', 'SVM (RBF)', 'LSTM'];

  // Bar Chart Data
  const chartData = {
    labels: modelNames.map(m => m === 'Gaussian HMM' ? 'Gaussian HMM (Proposed 🏆)' : m),
    datasets: [
      {
        label: 'Sensitivity (Recall)',
        data: modelNames.map(m => {
          const val = metricView === 'event' 
            ? models[m].event_metrics.sensitivity * 100 
            : models[m].window_metrics.sensitivity * 100;
          return Number(val.toFixed(2));
        }),
        backgroundColor: '#2563EB',
        borderRadius: 4,
      },
      {
        label: 'Specificity',
        data: modelNames.map(m => {
          const val = metricView === 'event' 
            ? models[m].event_metrics.specificity * 100 
            : models[m].window_metrics.specificity * 100;
          return Number(val.toFixed(2));
        }),
        backgroundColor: '#059669',
        borderRadius: 4,
      },
      {
        label: 'Precision',
        data: modelNames.map(m => {
          const val = metricView === 'event' 
            ? models[m].event_metrics.precision * 100 
            : models[m].window_metrics.precision * 100;
          return Number(val.toFixed(2));
        }),
        backgroundColor: '#7C3AED',
        borderRadius: 4,
      },
      {
        label: 'Fall F1-Score',
        data: modelNames.map(m => {
          const val = metricView === 'event' 
            ? models[m].event_metrics.f1_score * 100 
            : models[m].window_metrics.f1_score * 100;
          return Number(val.toFixed(2));
        }),
        backgroundColor: '#D97706',
        borderRadius: 4,
      },
      {
        label: 'Overall Accuracy',
        data: modelNames.map(m => {
          const val = models[m].overall_accuracy * 100;
          return Number(val.toFixed(2));
        }),
        backgroundColor: '#475569',
        borderRadius: 4,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { family: 'Inter', size: 12, weight: '500' },
          usePointStyle: true,
          boxWidth: 8,
          boxHeight: 8,
          padding: 18,
          color: '#334155'
        }
      },
      tooltip: {
        backgroundColor: '#0F172A',
        titleFont: { family: 'Inter', size: 13, weight: '600' },
        bodyFont: { family: 'Inter', size: 12 },
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          label: function(context) {
            return ` ${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        min: 70,
        max: 100,
        grid: { color: '#F1F5F9' },
        ticks: {
          font: { family: 'Inter', size: 11 },
          color: '#64748B',
          callback: (val) => `${val}%`
        },
        title: {
          display: true,
          text: 'Score (%)',
          color: '#64748B',
          font: { family: 'Inter', size: 12, weight: '600' }
        }
      },
      x: {
        grid: { display: false },
        ticks: {
          font: { family: 'Inter', size: 12, weight: '600' },
          color: '#1E293B'
        }
      }
    }
  };

  // Table Data Structure
  const tableRows = [
    {
      name: 'Gaussian HMM',
      role: 'Proposed Model',
      rank: 1,
      sens: models['Gaussian HMM'].event_metrics.sensitivity * 100,
      spec: models['Gaussian HMM'].event_metrics.specificity * 100,
      prec: models['Gaussian HMM'].event_metrics.precision * 100,
      f1: models['Gaussian HMM'].event_metrics.f1_score * 100,
      acc: models['Gaussian HMM'].overall_accuracy * 100,
      falseAlarm: models['Gaussian HMM'].event_metrics.false_alarms_per_hour,
      latency: models['Gaussian HMM'].inference_latency_ms_per_window
    },
    {
      name: 'LSTM Network',
      role: 'Comparative Model',
      rank: 2,
      sens: models['LSTM'].event_metrics.sensitivity * 100,
      spec: models['LSTM'].event_metrics.specificity * 100,
      prec: models['LSTM'].event_metrics.precision * 100,
      f1: models['LSTM'].event_metrics.f1_score * 100,
      acc: models['LSTM'].overall_accuracy * 100,
      falseAlarm: models['LSTM'].event_metrics.false_alarms_per_hour,
      latency: models['LSTM'].inference_latency_ms_per_window
    },
    {
      name: 'Random Forest',
      role: 'Comparative Model',
      rank: 3,
      sens: models['Random Forest'].event_metrics.sensitivity * 100,
      spec: models['Random Forest'].event_metrics.specificity * 100,
      prec: models['Random Forest'].event_metrics.precision * 100,
      f1: models['Random Forest'].event_metrics.f1_score * 100,
      acc: models['Random Forest'].overall_accuracy * 100,
      falseAlarm: models['Random Forest'].event_metrics.false_alarms_per_hour,
      latency: models['Random Forest'].inference_latency_ms_per_window
    },
    {
      name: 'SVM (RBF Kernel)',
      role: 'Comparative Model',
      rank: 4,
      sens: models['SVM (RBF)'].event_metrics.sensitivity * 100,
      spec: models['SVM (RBF)'].event_metrics.specificity * 100,
      prec: models['SVM (RBF)'].event_metrics.precision * 100,
      f1: models['SVM (RBF)'].event_metrics.f1_score * 100,
      acc: models['SVM (RBF)'].overall_accuracy * 100,
      falseAlarm: models['SVM (RBF)'].event_metrics.false_alarms_per_hour,
      latency: models['SVM (RBF)'].inference_latency_ms_per_window
    }
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Comprehensive Model Benchmark</h1>
        <p className="page-desc">
          Rigorous subject-independent evaluation across 327 test trials on held-out participants 
          (Adults SA20–SA23 and Elderly SE13–SE15). Comparing our Proposed Gaussian HMM with Viterbi decoding 
          against Random Forest, Support Vector Machine, and Long Short-Term Memory baselines.
        </p>
      </div>

      {/* Top Highlight Metric Cards */}
      <div className="stat-grid">
        <div className="stat-card winner">
          <div>
            <div className="stat-val primary">99.33%</div>
            <div className="stat-label">Fall Sensitivity (Recall)</div>
            <div className="stat-sub">Proposed HMM • 149 of 150 falls caught</div>
          </div>
          <div className="stat-icon" style={{ background: '#EFF6FF', color: '#2563EB' }}>
            <Award size={26} />
          </div>
        </div>

        <div className="stat-card winner">
          <div>
            <div className="stat-val primary">98.87%</div>
            <div className="stat-label">Fall Specificity</div>
            <div className="stat-sub">Only 2 false alarms in 177 ADLs</div>
          </div>
          <div className="stat-icon" style={{ background: '#ECFDF5', color: '#059669' }}>
            <CheckCircle size={26} />
          </div>
        </div>

        <div className="stat-card winner">
          <div>
            <div className="stat-val primary">99.00%</div>
            <div className="stat-label">Fall F1-Score</div>
            <div className="stat-sub">Proposed HMM Rank #1 Overall</div>
          </div>
          <div className="stat-icon" style={{ background: '#F5F3FF', color: '#7C3AED' }}>
            <TrendingUp size={26} />
          </div>
        </div>

        <div className="stat-card winner">
          <div>
            <div className="stat-val success">0.036 ms</div>
            <div className="stat-label">Inference Latency</div>
            <div className="stat-sub">46× faster than Random Forest</div>
          </div>
          <div className="stat-icon" style={{ background: '#FFFBEB', color: '#D97706' }}>
            <Zap size={26} />
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <TrendingUp size={20} color="#2563EB" />
              Comparative Multi-Metric Performance Overview
            </div>
            <div className="card-subtitle">
              Comparing Proposed Gaussian HMM against machine learning and deep learning baselines
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => setMetricView('event')}
              className={`badge ${metricView === 'event' ? 'badge-primary' : 'badge-neutral'}`}
              style={{ cursor: 'pointer', padding: '0.5rem 0.85rem' }}
            >
              Event-Level Metrics
            </button>
            <button
              onClick={() => setMetricView('window')}
              className={`badge ${metricView === 'window' ? 'badge-primary' : 'badge-neutral'}`}
              style={{ cursor: 'pointer', padding: '0.5rem 0.85rem' }}
            >
              Window-Level Metrics
            </button>
          </div>
        </div>

        <div className="chart-wrapper">
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Full Comparative Metrics Table */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Award size={20} color="#2563EB" />
              Full Comparative Benchmark Metrics Table
            </div>
            <div className="card-subtitle">
              Strict subject-independent holdout evaluation on 7 unseen subjects (327 trials)
            </div>
          </div>
          <span className="badge badge-success">Ranked by Overall Fall F1 & Accuracy</span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model Architecture</th>
                <th>Model Role</th>
                <th>Sensitivity (Recall)</th>
                <th>Specificity</th>
                <th>Precision</th>
                <th>Fall F1-Score</th>
                <th>Overall Accuracy</th>
                <th>False Alarms / Hr</th>
                <th>Inference Latency</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row) => (
                <tr key={row.name} className={row.rank === 1 ? 'highlight-row' : ''}>
                  <td>
                    {row.rank === 1 ? (
                      <span className="badge badge-primary">
                        <Award size={12} /> #1 Best
                      </span>
                    ) : (
                      <span className="badge badge-neutral">#{row.rank}</span>
                    )}
                  </td>
                  <td style={{ fontWeight: 600 }}>
                    {row.name}
                  </td>
                  <td>
                    {row.rank === 1 ? (
                      <span className="badge badge-primary">Proposed Model</span>
                    ) : (
                      <span className="badge badge-neutral">Comparative Model</span>
                    )}
                  </td>
                  <td style={{ fontWeight: row.rank === 1 ? '700' : '400', color: row.rank === 1 ? '#2563EB' : 'inherit' }}>
                    {row.sens.toFixed(2)}%
                  </td>
                  <td style={{ fontWeight: row.rank === 1 ? '700' : '400', color: row.rank === 1 ? '#059669' : 'inherit' }}>
                    {row.spec.toFixed(2)}%
                  </td>
                  <td style={{ fontWeight: row.rank === 1 ? '700' : '400', color: row.rank === 1 ? '#7C3AED' : 'inherit' }}>
                    {row.prec.toFixed(2)}%
                  </td>
                  <td style={{ fontWeight: row.rank === 1 ? '700' : '400', color: row.rank === 1 ? '#D97706' : 'inherit' }}>
                    {row.f1.toFixed(2)}%
                  </td>
                  <td style={{ fontWeight: row.rank === 1 ? '700' : '400' }}>
                    {row.acc.toFixed(2)}%
                  </td>
                  <td>
                    <span style={{ color: row.rank === 1 ? '#059669' : '#DC2626', fontWeight: 600 }}>
                      {row.falseAlarm.toFixed(2)} / hr
                    </span>
                  </td>
                  <td>
                    <code>{row.latency.toFixed(3)} ms</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrices Grid */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              Confusion Matrices (Event-Level Fall Detection)
            </div>
            <div className="card-subtitle">
              True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
            </div>
          </div>
        </div>

        <div className="cm-grid-container">
          {modelNames.map((name) => {
            const m = models[name];
            const cm = m.confusion_matrix_event;
            const tn = cm[0][0];
            const fp = cm[0][1];
            const fn = cm[1][0];
            const tp = cm[1][1];
            const isProposed = name === 'Gaussian HMM';

            return (
              <div 
                key={name} 
                className="cm-box"
                style={{ 
                  border: isProposed ? '2px solid #2563EB' : '1px solid var(--border-color)',
                  background: isProposed ? '#F8FAFF' : '#FFFFFF'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.95rem', color: isProposed ? '#2563EB' : '#1E293B' }}>
                    {name} {isProposed && '🏆 (Proposed)'}
                  </span>
                  <span className={`badge ${isProposed ? 'badge-primary' : 'badge-neutral'}`}>
                    F1: {(m.event_metrics.f1_score * 100).toFixed(1)}%
                  </span>
                </div>

                <table className="cm-table">
                  <tbody>
                    <tr>
                      <td style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>Actual ADL</td>
                      <td className="cm-cell cm-tn">
                        <div style={{ fontSize: '1.1rem' }}>{tn}</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>TN (Correct ADL)</div>
                      </td>
                      <td className="cm-cell cm-fp">
                        <div style={{ fontSize: '1.1rem' }}>{fp}</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>FP (False Alarm)</div>
                      </td>
                    </tr>
                    <tr>
                      <td style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>Actual Fall</td>
                      <td className="cm-cell cm-fn">
                        <div style={{ fontSize: '1.1rem' }}>{fn}</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>FN (Missed Fall)</div>
                      </td>
                      <td className="cm-cell cm-tp">
                        <div style={{ fontSize: '1.1rem' }}>{tp}</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>TP (Caught Fall)</div>
                      </td>
                    </tr>
                    <tr>
                      <td></td>
                      <td style={{ textAlign: 'center', fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>Pred ADL</td>
                      <td style={{ textAlign: 'center', fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>Pred Fall</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      </div>

      {/* Analytical Insights */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            Key Empirical Findings & Scientific Analysis
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.25rem' }}>
          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <h4 style={{ color: '#2563EB', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle size={18} />
              Temporal State Coherence
            </h4>
            <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
              Unlike static feature classifiers (RF, SVM), the Proposed Gaussian HMM explicitly models 
              the physiological transition from <code>Dynamic Activity → Fall Impact → Inactive Recovery</code>. 
              This temporal requirement rejects instantaneous spikes caused by jumping or fast sitting, reducing false alarms to 1.47/hr.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <h4 style={{ color: '#059669', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Zap size={18} />
              Real-Time Edge Efficiency
            </h4>
            <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
              Gaussian HMM inference runs via an optimized log-space Viterbi recursion taking only 
              <code>0.036 ms</code> per 2-second window. It is <strong>46× faster than Random Forest</strong> (1.67 ms) 
              and <strong>5.7× faster than LSTM</strong> (0.21 ms), making it ideal for microcontrollers and wearable sensors.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <h4 style={{ color: '#7C3AED', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={18} />
              Subject-Independent Robustness
            </h4>
            <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
              Evaluated strictly on 7 unseen subjects (SA20–SA23, SE13–SE15), the HMM maintains 99.33% recall 
              and 98.87% specificity across both young adult and elderly cohorts, avoiding subject-specific memorization traps.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
