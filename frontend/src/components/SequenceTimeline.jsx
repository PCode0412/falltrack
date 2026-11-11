import React, { useState } from 'react';
import { Layers, GitCommit, Check, X, Activity } from 'lucide-react';

const STATE_NAMES = ['Static Posture', 'Dynamic Activity', 'Postural Transition', 'Fall Impact', 'Post-Fall Recovery'];
const STATE_COLORS = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

// 5x5 Transition Matrix of Proposed Gaussian HMM
const TRANSITION_MATRIX = [
  [0.820, 0.050, 0.110, 0.010, 0.010], // Static
  [0.040, 0.860, 0.070, 0.025, 0.005], // Dynamic
  [0.120, 0.100, 0.650, 0.110, 0.020], // Transition
  [0.010, 0.010, 0.030, 0.100, 0.850], // Fall -> Recovery (high probability!)
  [0.150, 0.020, 0.030, 0.000, 0.800]  // Recovery -> Static
];

// Helper to collapse consecutive identical window states into clean continuous intervals
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

function TimelineBar({ segments }) {
  return (
    <div
      style={{
        display: 'flex',
        width: '100%',
        height: '38px',
        backgroundColor: '#F1F5F9',
        borderRadius: '8px',
        overflow: 'hidden',
        border: '1px solid #E2E8F0',
        position: 'relative',
      }}
    >
      {segments.map((seg, idx) => {
        // Show text label only if segment is wide enough to be readable
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
              fontSize: '0.78rem',
              fontWeight: 600,
              padding: '0 6px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              borderRight: idx < segments.length - 1 ? '1px solid rgba(255, 255, 255, 0.35)' : 'none',
              cursor: 'default',
              transition: 'filter 0.15s ease',
            }}
            title={`${seg.name}: ${seg.startTime}s – ${seg.endTime}s (${seg.count} window${seg.count > 1 ? 's' : ''})`}
          >
            {showLabel && (
              <span>
                {seg.name} <span style={{ opacity: 0.85, fontWeight: 400, fontSize: '0.72rem' }}>({seg.startTime}s–{seg.endTime}s)</span>
              </span>
            )}
            {showShort && (
              <span style={{ fontSize: '0.72rem' }}>
                {seg.name.split(' ')[0]}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function SequenceTimeline({ recordings = [] }) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (!recordings || recordings.length === 0) {
    return <div className="card">Loading sequence timeline data...</div>;
  }

  const currentRec = recordings[selectedIdx] || recordings[0] || {};
  const duration = currentRec.duration || 100;

  const yTrue = currentRec.y_true || [];
  const predsHMM = currentRec.preds_hmm || [];
  const predsRF = currentRec.preds_rf || [];
  const predsSVM = currentRec.preds_svm || [];
  const predsLSTM = currentRec.preds_lstm || [];

  const segsTrue = getContiguousSegments(yTrue, duration);
  const segsHMM = getContiguousSegments(predsHMM, duration);
  const segsLSTM = getContiguousSegments(predsLSTM, duration);
  const segsRF = getContiguousSegments(predsRF, duration);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Sequence Timeline & Viterbi Decoding</h1>
        <p className="page-desc">
          Window-by-window temporal decoding comparison between the Proposed Gaussian HMM 
          (optimal log-space Viterbi state path) and comparative frame-by-frame classifiers.
        </p>
      </div>

      {/* Recording Selector Bar */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: '1', minWidth: '320px' }}>
            <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Select Trial:
            </label>
            <select
              value={selectedIdx}
              onChange={(e) => setSelectedIdx(Number(e.target.value))}
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

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <span className={`badge ${currentRec.is_fall ? 'badge-danger' : 'badge-success'}`}>
              {currentRec.is_fall ? 'Ground Truth: Fall Event' : 'Ground Truth: ADL'}
            </span>
            <span className="badge badge-primary">
              Proposed HMM: {currentRec.fall_alert_hmm ? 'Fall Detected' : 'No Fall'}
            </span>
          </div>
        </div>
      </div>

      {/* Synchronized Multi-Model Timeline Strip */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Layers size={18} color="#2563EB" />
              Multi-Model Decoded Sequence Alignment
            </div>
            <div className="card-subtitle">
              Continuous state intervals across trial duration ({duration}s total)
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Ground Truth Strip */}
          <div>
            <div style={{ fontSize: '0.825rem', fontWeight: 700, color: '#334155', marginBottom: '0.4rem' }}>
              Ground Truth Sequence
            </div>
            <TimelineBar segments={segsTrue} />
          </div>

          {/* Proposed HMM Strip */}
          <div>
            <div style={{ fontSize: '0.825rem', fontWeight: 700, color: '#2563EB', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>Proposed Gaussian HMM (Viterbi) 🏆</span>
              <span className="badge badge-primary" style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem' }}>Optimal Path</span>
            </div>
            <div style={{ padding: '2px', background: '#DBEAFE', borderRadius: '10px' }}>
              <TimelineBar segments={segsHMM} />
            </div>
          </div>

          {/* LSTM Strip */}
          <div>
            <div style={{ fontSize: '0.825rem', fontWeight: 700, color: '#D97706', marginBottom: '0.4rem' }}>
              Comparative: LSTM Network
            </div>
            <TimelineBar segments={segsLSTM} />
          </div>

          {/* Random Forest Strip */}
          <div>
            <div style={{ fontSize: '0.825rem', fontWeight: 700, color: '#059669', marginBottom: '0.4rem' }}>
              Comparative: Random Forest (Window-Independent)
            </div>
            <TimelineBar segments={segsRF} />
          </div>

          {/* Time axis markers */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748B', borderTop: '1px dashed #CBD5E1', paddingTop: '0.4rem', paddingLeft: '4px', paddingRight: '4px' }}>
            <span>0.0s</span>
            <span>{(duration * 0.25).toFixed(1)}s</span>
            <span>{(duration * 0.50).toFixed(1)}s</span>
            <span>{(duration * 0.75).toFixed(1)}s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>

          {/* Legend */}
          <div className="timeline-legend" style={{ marginTop: '0.25rem' }}>
            {STATE_NAMES.map((name, idx) => (
              <div key={name} className="legend-item">
                <div className="legend-color" style={{ backgroundColor: STATE_COLORS[idx] }}></div>
                <span style={{ fontWeight: 500 }}>{name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Window-by-Window Alignment Table */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <GitCommit size={18} color="#2563EB" />
              Window-Level State Alignment Table
            </div>
            <div className="card-subtitle">
              Detailed breakdown of each 2.0s window with 50% overlap (step = 1.0s)
            </div>
          </div>
        </div>

        <div className="table-container" style={{ maxHeight: '360px', overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Window</th>
                <th>Time Interval</th>
                <th>True Annotation</th>
                <th>Proposed HMM</th>
                <th>LSTM Network</th>
                <th>Random Forest</th>
                <th>SVM (RBF)</th>
                <th>HMM Status</th>
              </tr>
            </thead>
            <tbody>
              {yTrue.map((trueState, idx) => {
                const hmmState = predsHMM[idx];
                const lstmState = predsLSTM[idx];
                const rfState = predsRF[idx];
                const svmState = predsSVM[idx];
                const isMatch = hmmState === trueState;
                const tStart = (idx * 1.0).toFixed(1);
                const tEnd = (idx * 1.0 + 2.0).toFixed(1);

                return (
                  <tr key={idx} className={isMatch ? '' : 'highlight-row'}>
                    <td style={{ fontWeight: 600 }}>W{idx + 1}</td>
                    <td><code>[{tStart}s – {tEnd}s]</code></td>
                    <td>
                      <span className="badge" style={{ backgroundColor: `${STATE_COLORS[trueState]}15`, color: STATE_COLORS[trueState], border: `1px solid ${STATE_COLORS[trueState]}40` }}>
                        {STATE_NAMES[trueState]}
                      </span>
                    </td>
                    <td>
                      <span className="badge" style={{ backgroundColor: `${STATE_COLORS[hmmState]}15`, color: STATE_COLORS[hmmState], border: `1px solid ${STATE_COLORS[hmmState]}40`, fontWeight: 700 }}>
                        {STATE_NAMES[hmmState]}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8rem', color: '#475569' }}>
                        {STATE_NAMES[lstmState]}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8rem', color: '#475569' }}>
                        {STATE_NAMES[rfState]}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8rem', color: '#475569' }}>
                        {STATE_NAMES[svmState]}
                      </span>
                    </td>
                    <td>
                      {isMatch ? (
                        <span className="badge badge-success" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem' }}>
                          <Check size={12} /> Correct
                        </span>
                      ) : (
                        <span className="badge badge-warning" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem' }}>
                          <X size={12} /> Deviation
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* HMM State Transition Matrix Heatmap */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <Activity size={18} color="#2563EB" />
              Proposed Gaussian HMM State Transition Matrix A [5 × 5]
            </div>
            <div className="card-subtitle">
              Learned transition probabilities {'A_ij = P(S_t = j | S_{t-1} = i)'} from Baum-Welch training
            </div>
          </div>
          <span className="badge badge-primary">Physiological Priors Preserved</span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th style={{ background: '#F1F5F9' }}>From State \ To State</th>
                {STATE_NAMES.map((name) => (
                  <th key={name} style={{ textAlign: 'center' }}>{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TRANSITION_MATRIX.map((row, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 700, backgroundColor: '#F8FAFC' }}>
                    {STATE_NAMES[i]}
                  </td>
                  {row.map((val, j) => {
                    const alpha = Math.min(val * 0.85 + 0.05, 1);
                    const bg = `rgba(37, 99, 235, ${alpha.toFixed(2)})`;

                    return (
                      <td
                        key={j}
                        style={{
                          textAlign: 'center',
                          backgroundColor: val > 0.05 ? bg : 'transparent',
                          color: val > 0.35 ? '#FFFFFF' : 'inherit',
                          fontWeight: val > 0.1 ? '700' : '400',
                          padding: '0.75rem'
                        }}
                      >
                        {(val * 100).toFixed(1)}%
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#64748B' }}>
          💡 <strong>Key Structural Observation:</strong> Notice the 85.0% transition probability from <em>Fall Impact → Post-Fall Recovery</em>. 
          A fall is only declared when an impact state transitions into sustained inactivity, eliminating false alarms from everyday sudden movements.
        </div>
      </div>
    </div>
  );
}
