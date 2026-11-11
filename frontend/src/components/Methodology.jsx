import React from 'react';
import { Cpu, Terminal, Layers, GitBranch } from 'lucide-react';

export default function Methodology() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">System Architecture & Mathematical Methodology</h1>
        <p className="page-desc">
          Formal mathematical formulation of the Proposed Gaussian Hidden Markov Model with log-space Viterbi decoding, 
          kinematic feature representations, and comparative baseline implementations.
        </p>
      </div>

      {/* Pipeline Stages Flow */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div className="card-title">
            <Cpu size={18} color="#2563EB" />
            End-to-End Processing Pipeline Architecture
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
          <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: '#2563EB', marginBottom: '0.4rem' }}>
              <span style={{ background: '#EFF6FF', width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem' }}>1</span>
              Acquisition & Filtering
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748B', lineHeight: 1.5 }}>
              Lumbar inertial stream sampled at 50 Hz. 4th-order zero-phase Butterworth filter (fc = 20 Hz) 
              removes high-frequency mechanical tremor and sensor quantization jitter.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: '#059669', marginBottom: '0.4rem' }}>
              <span style={{ background: '#ECFDF5', width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem' }}>2</span>
              Windowing (2.0s / 50%)
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748B', lineHeight: 1.5 }}>
              Continuous segmentation into W = 100 samples per window with 50-sample step size (1.0s overlap), 
              guaranteeing continuous temporal continuity across postural shifts.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: '#7C3AED', marginBottom: '0.4rem' }}>
              <span style={{ background: '#F5F3FF', width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem' }}>3</span>
              24-D Kinematic Features
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748B', lineHeight: 1.5 }}>
              12 acceleration features (mean, std, peak, RMS, zero-crossings, energy) and 12 angular velocity features 
              capturing 3D spatial rotational energy.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: '#D97706', marginBottom: '0.4rem' }}>
              <span style={{ background: '#FFFBEB', width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem' }}>4</span>
              Viterbi HMM Decoding
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748B', lineHeight: 1.5 }}>
              Log-space dynamic programming identifies global optimal latent state sequence Q*, 
              enforcing physiological transition rules to confirm true fall impacts.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Formulations */}
      <div className="grid-2col">
        {/* Gaussian HMM Formulation */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Layers size={18} color="#2563EB" />
              Gaussian HMM Mathematical Formulation
            </div>
          </div>

          <div style={{ fontSize: '0.875rem', color: '#334155', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <p>
              A Hidden Markov Model is specified by the parameter set {'\u03BB = (\u03C0, A, B)'}:
            </p>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>1. State Space:</strong><br />
              {'S = {S1: Static, S2: Dynamic, S3: Transition, S4: Fall Impact, S5: Recovery}'}
            </div>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>2. State Transition Matrix A:</strong><br />
              {'A_ij = P(q_t = S_j | q_{t-1} = S_i),  where  \u2211_j A_ij = 1'}
            </div>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>3. Continuous Gaussian Emission Distribution B:</strong><br />
              {'b_j(x) = (2\u03C0)^(-d/2) |\u03A3_j|^(-1/2) exp(-1/2 (x - \u03BC_j)^T \u03A3_j^(-1) (x - \u03BC_j))'}<br />
              {'where d = 24 kinematic features, \u03BC_j \u2208 \u211D\u00B2\u2074, \u03A3_j = diag(\u03C3_j1\u00B2, ..., \u03C3_j24\u00B2)'}
            </div>
          </div>
        </div>

        {/* Viterbi Decoding Algorithm */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Terminal size={18} color="#059669" />
              Log-Space Viterbi Path Optimization
            </div>
          </div>

          <div style={{ fontSize: '0.875rem', color: '#334155', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <p>
              To eliminate floating-point underflow over continuous streaming sequences, the Viterbi path 
              is computed in the logarithmic domain:
            </p>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>Initialization (t = 1):</strong><br />
              {'V_1(j) = ln(\u03C0_j) + ln(b_j(x_1)),  \u2200 j \u2208 {1..5}'}
            </div>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>Recursive Step (t = 2..T):</strong><br />
              {'V_t(j) = max_i [ V_{t-1}(i) + ln(A_ij) ] + ln(b_j(x_t))'}<br />
              {'\u03C8_t(j) = argmax_i [ V_{t-1}(i) + ln(A_ij) ]'}
            </div>

            <div style={{ background: '#F8FAFC', padding: '0.85rem', borderRadius: '6px', border: '1px solid #E2E8F0', fontFamily: 'JetBrains Mono', fontSize: '0.825rem' }}>
              <strong>Backtracking:</strong><br />
              {'q_T* = argmax_j [ V_T(j) ]'}<br />
              {'q_t* = \u03C8_{t+1}(q_{t+1}*),  for t = T-1 down to 1'}
            </div>
          </div>
        </div>
      </div>

      {/* Comparative Baseline Implementations */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <GitBranch size={18} color="#7C3AED" />
            Comparative Benchmark Models Implementation Details
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#059669', marginBottom: '0.4rem' }}>
              Random Forest Classifier
            </div>
            <ul style={{ fontSize: '0.85rem', color: '#475569', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
              <li><strong>Ensemble:</strong> 100 balanced decision trees</li>
              <li><strong>Splitting:</strong> Gini impurity criterion</li>
              <li><strong>Max Depth:</strong> 12 levels (tuned on validation split)</li>
              <li><strong>Class Weighting:</strong> Inverse frequency balanced</li>
              <li><strong>Limitation:</strong> Evaluates windows independently without temporal state sequence continuity.</li>
            </ul>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#7C3AED', marginBottom: '0.4rem' }}>
              Support Vector Machine (SVM)
            </div>
            <ul style={{ fontSize: '0.85rem', color: '#475569', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
              <li><strong>Kernel:</strong> Radial Basis Function (RBF)</li>
              <li><strong>Hyperparameters:</strong> Regularization C = 2.0, {'\u03B3 = 0.02'}</li>
              <li><strong>Standardization:</strong> Zero-mean, unit variance fitted strictly on training subjects</li>
              <li><strong>Margin:</strong> Soft-margin with balanced class weights</li>
              <li><strong>Limitation:</strong> Boundary overlap between abrupt sitting and stumble recovery creates false alarms.</li>
            </ul>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#D97706', marginBottom: '0.4rem' }}>
              Long Short-Term Memory (LSTM)
            </div>
            <ul style={{ fontSize: '0.85rem', color: '#475569', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
              <li><strong>Architecture:</strong> 2-layer Bidirectional LSTM</li>
              <li><strong>Hidden Units:</strong> 64 units per direction (128 total)</li>
              <li><strong>Regularization:</strong> Dropout = 0.30 between recurrent layers</li>
              <li><strong>Optimizer:</strong> Adam (lr = 0.001, weight decay = 0.0001)</li>
              <li><strong>Limitation:</strong> Higher parameter footprint and latency (0.21 ms vs 0.036 ms for HMM).</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
