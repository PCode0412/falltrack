import React, { useState, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Activity, CheckCircle2, Play, Zap, ShieldAlert, User, Clock, Compass } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const STATE_NAMES = ['Static Posture', 'Dynamic Activity', 'Postural Transition', 'Fall Impact', 'Post-Fall Recovery'];
const STATE_COLORS = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

function getContiguousSegments(states, totalDuration = 100) {
  if (!states || states.length === 0) return [];
  const segments = [];
  let current = { state: states[0], startIdx: 0, count: 1 };

  for (let i = 1; i < states.length; i++) {
    if (states[i] === current.state) {
      current.count++;
    } else {
      segments.push(current);
      current = { state: states[i], startIdx: i, count: 1 };
    }
  }
  segments.push(current);

  const total = states.length;
  return segments.map((seg) => {
    const widthPct = (seg.count / total) * 100;
    const startTime = (seg.startIdx * 1.0).toFixed(1);
    const endTime = Math.min(totalDuration, seg.startIdx * 1.0 + seg.count * 1.0 + 1.0).toFixed(1);
    const name = STATE_NAMES[seg.state] || 'Unknown';
    const color = STATE_COLORS[seg.state] || '#94A3B8';

    return {
      state: seg.state,
      count: seg.count,
      widthPct,
      startTime,
      endTime,
      name,
      color,
    };
  });
}

export default function SensorStream({ recordings = [] }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [filterType, setFilterType] = useState('all'); // 'all', 'fall', 'adl'
  const [selectedDecoder, setSelectedDecoder] = useState('preds_hmm'); // 'y_true', 'preds_hmm', 'preds_rf', 'preds_svm', 'preds_lstm'
  const [alertDismissed, setAlertDismissed] = useState(false);

  // Filter recordings
  const filteredRecordings = useMemo(() => {
    if (!recordings || recordings.length === 0) return [];
    if (filterType === 'fall') return recordings.filter(r => r.is_fall);
    if (filterType === 'adl') return recordings.filter(r => !r.is_fall);
    return recordings;
  }, [recordings, filterType]);

  if (!recordings || recordings.length === 0) {
    return <div className="card">Loading sensor recordings...</div>;
  }

  const currentRec = recordings[selectedIndex] || recordings[0] || {};
  const duration = currentRec.duration || 100;

  // Downsample for chart display so lines render cleanly and smoothly
  const timeArray = currentRec.time || [];
  const step = Math.max(1, Math.floor(timeArray.length / 250));
  const downsampledTime = [];
  const downsampledAccX = [];
  const downsampledAccY = [];
  const downsampledAccZ = [];
  const downsampledAccMag = [];
  const downsampledGyroMag = [];

  for (let i = 0; i < timeArray.length; i += step) {
    downsampledTime.push(timeArray[i].toFixed(2) + 's');
    downsampledAccX.push(currentRec.acc_x?.[i] ?? 0);
    downsampledAccY.push(currentRec.acc_y?.[i] ?? 0);
    downsampledAccZ.push(currentRec.acc_z?.[i] ?? 0);
    downsampledAccMag.push(currentRec.acc_mag?.[i] ?? 0);
    downsampledGyroMag.push(currentRec.gyro_mag?.[i] ?? 0);
  }

  // Chart configuration for Accelerometer
  const accChartData = {
    labels: downsampledTime,
    datasets: [
      {
        label: 'Magnitude |a| (g)',
        data: downsampledAccMag,
        borderColor: '#2563EB',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      },
      {
        label: 'Acc X (Lateral)',
        data: downsampledAccX,
        borderColor: '#94A3B8',
        borderWidth: 1.2,
        pointRadius: 0,
        borderDash: [3, 3],
      },
      {
        label: 'Acc Y (Vertical)',
        data: downsampledAccY,
        borderColor: '#64748B',
        borderWidth: 1.2,
        pointRadius: 0,
        borderDash: [3, 3],
      },
      {
        label: 'Acc Z (Anterior-Posterior)',
        data: downsampledAccZ,
        borderColor: '#475569',
        borderWidth: 1.2,
        pointRadius: 0,
        borderDash: [3, 3],
      },
      {
        label: 'Impact Threshold (2.8g)',
        data: Array(downsampledTime.length).fill(2.8),
        borderColor: '#DC2626',
        borderWidth: 1.5,
        borderDash: [6, 4],
        pointRadius: 0,
      }
    ]
  };

  const gyroChartData = {
    labels: downsampledTime,
    datasets: [
      {
        label: 'Angular Velocity |\u03C9| (\u00B0/s)',
        data: downsampledGyroMag,
        borderColor: '#7C3AED',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { family: 'Inter', size: 11, weight: '500' },
          boxWidth: 8,
          boxHeight: 8,
          usePointStyle: true,
          padding: 14,
          color: '#334155'
        }
      },
      tooltip: {
        backgroundColor: '#0F172A',
        titleFont: { family: 'Inter', size: 12, weight: '600' },
        bodyFont: { family: 'Inter', size: 11 },
        padding: 10,
        cornerRadius: 6,
      }
    },
    scales: {
      y: {
        grid: { color: '#F1F5F9' },
        ticks: { font: { family: 'Inter', size: 10 }, color: '#64748B' },
      },
      x: {
        grid: { display: false },
        ticks: {
          font: { family: 'Inter', size: 10 },
          color: '#64748B',
          maxTicksLimit: 12,
        }
      }
    }
  };

  // Timeline decoding array
  const activePreds = currentRec[selectedDecoder] || currentRec.preds_hmm || [];
  const activeSegments = getContiguousSegments(activePreds, duration);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Live Sensor Stream & Fall Alert System</h1>
        <p className="page-desc">
          Continuous dual-triaxial kinematic streams (ADXL345 accelerometer & ITG3200 gyroscope) 
          sampled at 50 Hz from a waist-mounted sensor. Real-time sequence decoding and instant fall impact classification.
        </p>
      </div>

      {/* Alert Banner */}
      {currentRec.is_fall && !alertDismissed ? (
        <div className="alert-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ background: '#FEE2E2', padding: '0.75rem', borderRadius: '50%', color: '#DC2626' }}>
              <ShieldAlert size={28} />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#991B1B' }}>
                CRITICAL ALERT: Fall Event Confirmed by Proposed Gaussian HMM
              </div>
              <div style={{ fontSize: '0.875rem', color: '#B91C1C', marginTop: '0.2rem' }}>
                Recording <strong>{currentRec.filename}</strong> ({currentRec.activity_name}) • 
                Peak Impact: <strong>{(currentRec.peak_acc_g ?? 0).toFixed(2)}g</strong> • 
                Subject: <strong>{currentRec.subject}</strong> • State: Impact followed by Inactive Recovery
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button 
              onClick={() => setAlertDismissed(true)} 
              className="badge badge-danger" 
              style={{ cursor: 'pointer', padding: '0.6rem 1.2rem', fontSize: '0.85rem' }}
            >
              Acknowledge Alert
            </button>
          </div>
        </div>
      ) : (
        <div className="alert-banner-safe">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ background: '#D1FAE5', padding: '0.75rem', borderRadius: '50%', color: '#059669' }}>
              <CheckCircle2 size={28} />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: '#065F46' }}>
                Normal Activity Stream • No Fall Anomalies Detected
              </div>
              <div style={{ fontSize: '0.85rem', color: '#047857', marginTop: '0.2rem' }}>
                Recording <strong>{currentRec.filename}</strong> ({currentRec.activity_name}) • 
                Peak Acceleration: <strong>{(currentRec.peak_acc_g ?? 0).toFixed(2)}g</strong> • Subject: <strong>{currentRec.subject}</strong>
              </div>
            </div>
          </div>
          <span className="badge badge-success">System Stable</span>
        </div>
      )}

      {/* Recording Selector & Filter Bar */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: '1', minWidth: '320px' }}>
            <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Select Recording:
            </label>
            <select
              value={selectedIndex}
              onChange={(e) => {
                setSelectedIndex(Number(e.target.value));
                setAlertDismissed(false);
              }}
              style={{
                flex: 1,
                padding: '0.5rem 0.85rem',
                borderRadius: '6px',
                border: '1px solid var(--border-color)',
                fontSize: '0.875rem',
                backgroundColor: '#FFFFFF',
                color: 'var(--text-primary)',
                fontWeight: 500,
                cursor: 'pointer'
              }}
            >
              {recordings.map((rec, idx) => (
                <option key={rec.filename} value={idx}>
                  [{rec.is_fall ? 'FALL ⚠️' : 'ADL'}] {rec.filename} - {rec.activity_name} ({rec.subject})
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Filter:</span>
            <button
              onClick={() => setFilterType('all')}
              className={`badge ${filterType === 'all' ? 'badge-primary' : 'badge-neutral'}`}
              style={{ cursor: 'pointer', padding: '0.4rem 0.75rem' }}
            >
              All Trials
            </button>
            <button
              onClick={() => setFilterType('fall')}
              className={`badge ${filterType === 'fall' ? 'badge-danger' : 'badge-neutral'}`}
              style={{ cursor: 'pointer', padding: '0.4rem 0.75rem' }}
            >
              Falls Only
            </button>
            <button
              onClick={() => setFilterType('adl')}
              className={`badge ${filterType === 'adl' ? 'badge-success' : 'badge-neutral'}`}
              style={{ cursor: 'pointer', padding: '0.4rem 0.75rem' }}
            >
              ADLs Only
            </button>
          </div>
        </div>

        {/* Recording Stats Chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#475569' }}>
            <User size={15} color="#2563EB" />
            <span>Participant: <strong>{currentRec.subject}</strong></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#475569' }}>
            <Clock size={15} color="#059669" />
            <span>Duration: <strong>{currentRec.duration}s</strong></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#475569' }}>
            <Zap size={15} color="#D97706" />
            <span>Peak Acc: <strong>{(currentRec.peak_acc_g ?? 0).toFixed(2)}g</strong></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#475569' }}>
            <Activity size={15} color="#7C3AED" />
            <span>Sampling: <strong>50 Hz (20 Hz Low-Pass Filtered)</strong></span>
          </div>
        </div>
      </div>

      {/* Neatly Spaced Sensor Signal Graphs */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Activity size={18} color="#2563EB" />
              Triaxial Accelerometer Stream & Magnitude [g]
            </div>
            <div className="card-subtitle">
              Low-pass filtered kinematic signals (fc = 20 Hz, 4th-order zero-phase Butterworth)
            </div>
          </div>
          <span className="badge badge-primary">Impact Marker: 2.80g</span>
        </div>
        <div className="signal-chart-wrapper">
          <Line data={accChartData} options={chartOptions} />
        </div>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Compass size={18} color="#7C3AED" />
              Triaxial Gyroscope Angular Velocity Magnitude [°/s]
            </div>
            <div className="card-subtitle">
              Rotational rate dynamics capturing trunk orientation shifts
            </div>
          </div>
        </div>
        <div className="signal-chart-wrapper">
          <Line data={gyroChartData} options={chartOptions} />
        </div>
      </div>

      {/* Spacious Decoded Sequence Timeline */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Play size={18} color="#059669" />
              Decoded Sequence Timeline & Temporal States
            </div>
            <div className="card-subtitle">
              Continuous state intervals across trial duration ({duration}s total)
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Decoder:</span>
            <select
              value={selectedDecoder}
              onChange={(e) => setSelectedDecoder(e.target.value)}
              style={{
                padding: '0.4rem 0.75rem',
                borderRadius: '6px',
                border: '1px solid var(--border-color)',
                fontSize: '0.8rem',
                backgroundColor: '#FFFFFF',
                fontWeight: 600,
                color: '#2563EB',
                cursor: 'pointer'
              }}
            >
              <option value="preds_hmm">Proposed Gaussian HMM (Viterbi) 🏆</option>
              <option value="y_true">Ground Truth Annotation</option>
              <option value="preds_lstm">Comparative: LSTM Network</option>
              <option value="preds_rf">Comparative: Random Forest</option>
              <option value="preds_svm">Comparative: Support Vector Machine</option>
            </select>
          </div>
        </div>

        {/* Spacious Continuous Timeline Bar */}
        <div style={{ padding: '0.5rem 0' }}>
          <div
            style={{
              display: 'flex',
              width: '100%',
              height: '42px',
              backgroundColor: '#F1F5F9',
              borderRadius: '8px',
              overflow: 'hidden',
              border: '1px solid #E2E8F0',
            }}
          >
            {activeSegments.map((seg, idx) => {
              const showLabel = seg.widthPct >= 8;
              const showShort = seg.widthPct >= 4 && !showLabel;

              return (
                <div
                  key={idx}
                  style={{
                    width: `${seg.widthPct}%`,
                    backgroundColor: seg.color,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#FFFFFF',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    padding: '0 6px',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    borderRight: idx < activeSegments.length - 1 ? '1px solid rgba(255, 255, 255, 0.35)' : 'none',
                    cursor: 'default',
                  }}
                  title={`${seg.name}: ${seg.startTime}s – ${seg.endTime}s (${seg.count} window${seg.count > 1 ? 's' : ''})`}
                >
                  {showLabel && (
                    <span>
                      {seg.name} <span style={{ opacity: 0.85, fontWeight: 400, fontSize: '0.72rem' }}>({seg.startTime}s–{seg.endTime}s)</span>
                    </span>
                  )}
                  {showShort && (
                    <span style={{ fontSize: '0.75rem' }}>{seg.name.split(' ')[0]}</span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Time axis labels */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748B', marginTop: '0.4rem', borderTop: '1px dashed #CBD5E1', paddingTop: '0.3rem', paddingLeft: '4px', paddingRight: '4px' }}>
            <span>0.0s</span>
            <span>{(duration * 0.25).toFixed(1)}s</span>
            <span>{(duration * 0.50).toFixed(1)}s</span>
            <span>{(duration * 0.75).toFixed(1)}s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>

          {/* Timeline Legend */}
          <div className="timeline-legend" style={{ marginTop: '0.5rem' }}>
            {STATE_NAMES.map((name, idx) => (
              <div key={name} className="legend-item">
                <div className="legend-color" style={{ backgroundColor: STATE_COLORS[idx] }}></div>
                <span style={{ fontWeight: 500 }}>{name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
