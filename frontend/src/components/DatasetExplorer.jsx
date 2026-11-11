import React, { useState, useMemo } from 'react';
import { Search, Users, Activity, HardDrive, ShieldCheck } from 'lucide-react';

export default function DatasetExplorer({ subjects = [], activities = [] }) {
  const [subSearch, setSubSearch] = useState('');
  const [subGroupFilter, setSubGroupFilter] = useState('all'); // 'all', 'Adult', 'Elderly'
  const [subSplitFilter, setSubSplitFilter] = useState('all'); // 'all', 'Training', 'Test'

  const [actSearch, setActSearch] = useState('');
  const [actTypeFilter, setActTypeFilter] = useState('all'); // 'all', 'ADL', 'Fall'

  // Filtered Subjects
  const filteredSubjects = useMemo(() => {
    if (!subjects) return [];
    return subjects.filter((s) => {
      const matchSearch = s.id ? s.id.toLowerCase().includes(subSearch.toLowerCase()) : false;
      const matchGroup = subGroupFilter === 'all' || s.group === subGroupFilter;
      const matchSplit = subSplitFilter === 'all' || s.split === subSplitFilter;
      return matchSearch && matchGroup && matchSplit;
    });
  }, [subjects, subSearch, subGroupFilter, subSplitFilter]);

  // Filtered Activities
  const filteredActivities = useMemo(() => {
    if (!activities) return [];
    return activities.filter((a) => {
      const matchSearch = (a.name && a.name.toLowerCase().includes(actSearch.toLowerCase())) || 
                          (a.code && a.code.toLowerCase().includes(actSearch.toLowerCase()));
      const matchType = actTypeFilter === 'all' || a.type === actTypeFilter;
      return matchSearch && matchType;
    });
  }, [activities, actSearch, actTypeFilter]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">SisFall Dataset Explorer & Protocol Directory</h1>
        <p className="page-desc">
          Official SisFall public benchmark dataset consisting of 4,505 continuous recordings collected from 
          38 participants performing 34 standardized activity protocols with waist-worn inertial sensors.
        </p>
      </div>

      {/* Dataset Overview Stat Grid */}
      <div className="stat-grid">
        <div className="stat-card">
          <div>
            <div className="stat-val primary">38</div>
            <div className="stat-label">Total Participants</div>
            <div className="stat-sub">23 Young Adults + 15 Elderly</div>
          </div>
          <div className="stat-icon" style={{ background: '#EFF6FF', color: '#2563EB' }}>
            <Users size={26} />
          </div>
        </div>

        <div className="stat-card">
          <div>
            <div className="stat-val primary">34</div>
            <div className="stat-label">Activity Protocols</div>
            <div className="stat-sub">19 ADLs + 15 Controlled Falls</div>
          </div>
          <div className="stat-icon" style={{ background: '#ECFDF5', color: '#059669' }}>
            <Activity size={26} />
          </div>
        </div>

        <div className="stat-card">
          <div>
            <div className="stat-val primary">4,505</div>
            <div className="stat-label">Sensor Trials</div>
            <div className="stat-sub">1,798 Falls + 2,707 ADLs</div>
          </div>
          <div className="stat-icon" style={{ background: '#F5F3FF', color: '#7C3AED' }}>
            <HardDrive size={26} />
          </div>
        </div>

        <div className="stat-card winner">
          <div>
            <div className="stat-val primary">7 Held-Out</div>
            <div className="stat-label">Test Evaluation Cohort</div>
            <div className="stat-sub">SA20–SA23 & SE13–SE15 (327 trials)</div>
          </div>
          <div className="stat-icon" style={{ background: '#FFFBEB', color: '#D97706' }}>
            <ShieldCheck size={26} />
          </div>
        </div>
      </div>

      {/* Two Column Section: Subjects & Activities */}
      <div className="grid-2col">
        {/* Subjects Table */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Users size={18} color="#2563EB" />
                Participant Demographics (38 Subjects)
              </div>
              <div className="card-subtitle">
                Filtered view: {filteredSubjects.length} of {subjects.length} participants
              </div>
            </div>
          </div>

          {/* Filters */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
            <div style={{ position: 'relative', flex: 1, minWidth: '140px' }}>
              <input
                type="text"
                placeholder="Search Subject ID..."
                value={subSearch}
                onChange={(e) => setSubSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.4rem 0.6rem 0.4rem 2rem',
                  fontSize: '0.8rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                }}
              />
              <Search size={14} style={{ position: 'absolute', left: '8px', top: '10px', color: '#94A3B8' }} />
            </div>

            <select
              value={subGroupFilter}
              onChange={(e) => setSubGroupFilter(e.target.value)}
              style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}
            >
              <option value="all">All Cohorts</option>
              <option value="Adult">Young Adults (19–30)</option>
              <option value="Elderly">Elderly Adults (60–75)</option>
            </select>

            <select
              value={subSplitFilter}
              onChange={(e) => setSubSplitFilter(e.target.value)}
              style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}
            >
              <option value="all">All Splits</option>
              <option value="Training">Training Set</option>
              <option value="Test">Held-out Test</option>
            </select>
          </div>

          <div className="table-container" style={{ maxHeight: '420px', overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Subject</th>
                  <th>Cohort</th>
                  <th>Age</th>
                  <th>Height</th>
                  <th>Weight</th>
                  <th>Gender</th>
                  <th>Split</th>
                </tr>
              </thead>
              <tbody>
                {filteredSubjects.map((s) => (
                  <tr key={s.id} className={s.split === 'Test' ? 'highlight-row' : ''}>
                    <td style={{ fontWeight: 700 }}>{s.id}</td>
                    <td>
                      <span className={`badge ${s.group === 'Elderly' ? 'badge-warning' : 'badge-primary'}`}>
                        {s.group}
                      </span>
                    </td>
                    <td>{s.age} yrs</td>
                    <td>{s.height} cm</td>
                    <td>{s.weight} kg</td>
                    <td>{s.gender}</td>
                    <td>
                      <span className={`badge ${s.split === 'Test' ? 'badge-danger' : 'badge-neutral'}`}>
                        {s.split}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Activity Protocols Table */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Activity size={18} color="#059669" />
                Activity Protocols (34 Protocols)
              </div>
              <div className="card-subtitle">
                Filtered view: {filteredActivities.length} of {activities.length} protocols
              </div>
            </div>
          </div>

          {/* Filters */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
            <div style={{ position: 'relative', flex: 1, minWidth: '140px' }}>
              <input
                type="text"
                placeholder="Search Activity..."
                value={actSearch}
                onChange={(e) => setActSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.4rem 0.6rem 0.4rem 2rem',
                  fontSize: '0.8rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                }}
              />
              <Search size={14} style={{ position: 'absolute', left: '8px', top: '10px', color: '#94A3B8' }} />
            </div>

            <select
              value={actTypeFilter}
              onChange={(e) => setActTypeFilter(e.target.value)}
              style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}
            >
              <option value="all">All Types</option>
              <option value="ADL">ADLs Only (D01–D19)</option>
              <option value="Fall">Falls Only (F01–F15)</option>
            </select>
          </div>

          <div className="table-container" style={{ maxHeight: '420px', overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Protocol Name</th>
                  <th>Type</th>
                  <th>Category</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {filteredActivities.map((a) => (
                  <tr key={a.code} className={a.type === 'Fall' ? 'highlight-row' : ''}>
                    <td style={{ fontWeight: 700 }}><code>{a.code}</code></td>
                    <td style={{ fontWeight: 500 }}>{a.name}</td>
                    <td>
                      <span className={`badge ${a.type === 'Fall' ? 'badge-danger' : 'badge-success'}`}>
                        {a.type}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-neutral">{a.category}</span>
                    </td>
                    <td>{a.duration_sec}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Sensor Specifications Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <HardDrive size={18} color="#2563EB" />
            Hardware & Inertial Sensor Specifications
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#1E293B', marginBottom: '0.35rem' }}>
              Primary Accelerometer (ADXL345)
            </div>
            <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
              Triaxial MEMS accelerometer configured for ±16g dynamic range with 13-bit resolution (3.9 mg/LSB). 
              Primary source for detecting sudden high-g impacts during fall collisions.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#1E293B', marginBottom: '0.35rem' }}>
              Triaxial Gyroscope (ITG-3200)
            </div>
            <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
              Angular rate sensor with ±2000 deg/s full-scale range and 16-bit analog-to-digital converters. 
              Crucial for tracking torso rotation, pitch/roll tilt, and distinguishes dynamic stumbling from intentional lying down.
            </p>
          </div>

          <div style={{ background: '#F8FAFC', padding: '1.25rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontWeight: 700, color: '#1E293B', marginBottom: '0.35rem' }}>
              Sensor Placement & Sampling
            </div>
            <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
              Fixed to the subject's waist (lumbar spine at L4–L5 level) using a firm belt buckle. 
              Raw acquisition at 200 Hz, downsampled to 50 Hz and preprocessed with a 20 Hz low-pass filter to remove tremor noise.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
