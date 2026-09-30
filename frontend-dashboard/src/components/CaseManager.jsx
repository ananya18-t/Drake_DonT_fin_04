/**
 * @file Alert queue drawer: search/filter alerts, pick the active case, assign a reviewer.
 */

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Inbox, ListFilter, Loader2, RefreshCw, Search, UserCheck, X } from 'lucide-react';
import { RiskBadge } from './EvidencePanel';
import { formatRelative } from '../utils/format';

const RISK_FILTERS = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

function QueueSkeleton() {
  return (
    <div className="space-y-2 p-3" role="status" aria-label="Loading alerts">
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="skeleton h-[68px]" />
      ))}
    </div>
  );
}

/**
 * @param {{
 *   open: boolean,
 *   onClose: () => void,
 *   alerts: import('../api/client').Alert[],
 *   loading?: boolean,
 *   error?: Error | null,
 *   onRetry?: () => void,
 *   selectedAlertId?: string | null,
 *   onSelectAlert: (alertId: string) => void,
 *   reviewers?: import('../api/client').Reviewer[],
 *   reviewerName?: (id: string|null) => string,
 *   onAssign: (caseId: string, reviewerId: string) => void,
 *   assigningCaseId?: string | null,
 * }} props
 */
export default function CaseManager({
  open,
  onClose,
  alerts,
  loading = false,
  error = null,
  onRetry,
  selectedAlertId = null,
  onSelectAlert,
  reviewers = [],
  reviewerName = (id) => id ?? 'Unassigned',
  onAssign,
  assigningCaseId = null,
}) {
  const [query, setQuery] = useState('');
  const [risk, setRisk] = useState('ALL');
  const [reviewerId, setReviewerId] = useState('');

  const selected = alerts.find((a) => a.alert_id === selectedAlertId) ?? null;

  useEffect(() => {
    setReviewerId(selected?.assigned_to ?? '');
  }, [selected?.alert_id, selected?.assigned_to]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return alerts.filter(
      (a) =>
        (risk === 'ALL' || a.risk_level === risk) &&
        (!q ||
          [a.alert_id, a.case_id, a.title, a.emp_id, a.employee_name].some((f) => f?.toLowerCase().includes(q))),
    );
  }, [alerts, query, risk]);

  if (!open) return null;

  const assigning = selected && assigningCaseId === selected.case_id;
  const canAssign = selected && reviewerId && reviewerId !== selected.assigned_to && !assigning;

  return (
    <div className="fixed inset-0 z-40 flex" role="dialog" aria-modal="true" aria-label="Alert queue">
      <aside className="relative flex h-full w-full max-w-md animate-drawer-in flex-col border-r border-slate-800 bg-slate-950 shadow-2xl">
        <header className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">Alert Queue</h2>
            <p className="text-[11px] text-slate-500">
              {alerts.length} alerts · {alerts.filter((a) => a.status === 'OPEN').length} unassigned
            </p>
          </div>
          <div className="flex items-center gap-1">
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                disabled={loading}
                className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
                aria-label="Refresh alerts"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              aria-label="Close alert queue"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        <div className="space-y-2 border-b border-slate-800 px-4 py-3">
          <label className="relative block">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" aria-hidden />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search alert, case, employee…"
              className="w-full rounded-lg border border-slate-800 bg-slate-900 py-1.5 pl-8 pr-3 text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500/60"
            />
          </label>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by risk">
            <ListFilter className="h-3.5 w-3.5 text-slate-500" aria-hidden />
            {RISK_FILTERS.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRisk(r)}
                aria-pressed={risk === r}
                className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide transition ${
                  risk === r ? 'bg-slate-200 text-slate-900' : 'bg-slate-800/70 text-slate-400 hover:text-slate-200'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {loading && alerts.length === 0 ? (
            <QueueSkeleton />
          ) : error && alerts.length === 0 ? (
            <div className="flex flex-col items-center p-8 text-center">
              <AlertTriangle className="mb-2 h-6 w-6 text-red-400" aria-hidden />
              <p className="text-sm text-slate-200">Could not load alerts</p>
              <p className="mt-1 text-xs text-slate-500">{error.message}</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center p-8 text-center">
              <Inbox className="mb-2 h-6 w-6 text-slate-600" aria-hidden />
              <p className="text-sm text-slate-300">{alerts.length === 0 ? 'Queue is clear' : 'No alerts match'}</p>
            </div>
          ) : (
            <ul className="space-y-1.5 p-3">
              {filtered.map((a) => {
                const isSelected = a.alert_id === selectedAlertId;
                return (
                  <li key={a.alert_id}>
                    <button
                      type="button"
                      onClick={() => onSelectAlert(a.alert_id)}
                      aria-current={isSelected}
                      className={`w-full rounded-lg border px-3 py-2.5 text-left transition ${
                        isSelected ? 'border-sky-500/60 bg-sky-500/10' : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <RiskBadge level={a.risk_level} size="sm" />
                        <span className="text-[10px] text-slate-500">{formatRelative(a.created_at)}</span>
                      </div>
                      <p className="mt-1.5 line-clamp-1 text-sm font-medium text-slate-100">{a.title}</p>
                      <div className="mt-1 flex items-center justify-between gap-2 text-[11px] text-slate-500">
                        <span className="truncate">
                          <span className="font-mono">{a.alert_id}</span> · {a.employee_name ?? a.emp_id}
                        </span>
                        <span className="shrink-0">
                          <span className="font-semibold tabular-nums text-slate-300">{a.composite_score}</span> ·{' '}
                          {a.assigned_to ? reviewerName(a.assigned_to) : a.status}
                        </span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {selected && (
          <footer className="space-y-2 border-t border-slate-800 bg-slate-900/80 p-4">
            <p className="panel-title flex items-center gap-1.5">
              <UserCheck className="h-3.5 w-3.5" aria-hidden /> Assign {selected.case_id}
            </p>
            <div className="flex gap-2">
              <select
                value={reviewerId}
                onChange={(e) => setReviewerId(e.target.value)}
                disabled={reviewers.length === 0 || assigning}
                className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-200 disabled:opacity-50"
                aria-label="Reviewer"
              >
                <option value="">{reviewers.length ? 'Select reviewer…' : 'No reviewers available'}</option>
                {reviewers.map((r) => (
                  <option key={r.reviewer_id} value={r.reviewer_id}>
                    {r.name} — {r.role}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => onAssign(selected.case_id, reviewerId)}
                disabled={!canAssign}
                className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500 px-3 py-1.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {assigning && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
                Assign
              </button>
            </div>
          </footer>
        )}
      </aside>
      <button type="button" className="flex-1 bg-slate-950/60 backdrop-blur-sm" onClick={onClose} aria-label="Close alert queue" />
    </div>
  );
}
