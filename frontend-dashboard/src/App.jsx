/**
 * @file Investigator dashboard shell.
 *
 * Layout: top bar (metrics + Tailscale node health) · left 60% (graph above timeline)
 * · right 40% (evidence panel). Owns application state: alert queue, active alert,
 * investigation bundle, graph/timeline selection, case actions and toasts.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertOctagon,
  BrainCircuit,
  Briefcase,
  CheckCircle2,
  Cpu,
  Database,
  Info,
  ListChecks,
  RefreshCw,
  Server,
  Shield,
  Users,
  Activity,
  X,
  XCircle,
} from 'lucide-react';
import {
  API_CONFIG,
  assignCase,
  escalateCase,
  exportEvidenceDossier,
  fetchAlerts,
  fetchReviewers,
  fetchSystemStatus,
  freezeAccounts,
  getErrorMessage,
  getInvestigationDetails,
} from './api/client';
import GraphCanvas from './components/GraphCanvas';
import TimelineStream from './components/TimelineStream';
import EvidencePanel from './components/EvidencePanel';
import CaseManager from './components/CaseManager';
import { formatNumber } from './utils/format';

const STATUS_POLL_MS = 15_000;
const TOAST_TTL_MS = 4_500;

// ─────────────────────────────────────────────────────────────── Toasts

const TOAST_TONES = {
  success: { Icon: CheckCircle2, className: 'border-emerald-500/40 text-emerald-300' },
  error: { Icon: XCircle, className: 'border-red-500/40 text-red-300' },
  info: { Icon: Info, className: 'border-sky-500/40 text-sky-300' },
};

function useToasts() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    clearTimeout(timers.current.get(id));
    timers.current.delete(id);
  }, []);

  const pushToast = useCallback(
    ({ tone = 'info', title, message }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      setToasts((prev) => [...prev.slice(-3), { id, tone, title, message }]);
      timers.current.set(id, setTimeout(() => dismissToast(id), TOAST_TTL_MS));
    },
    [dismissToast],
  );

  useEffect(() => {
    const map = timers.current;
    return () => map.forEach((t) => clearTimeout(t));
  }, []);

  return { toasts, pushToast, dismissToast };
}

function Toaster({ toasts, onDismiss }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[min(360px,calc(100vw-2rem))] flex-col gap-2" aria-live="polite">
      {toasts.map((t) => {
        const { Icon, className } = TOAST_TONES[t.tone] ?? TOAST_TONES.info;
        return (
          <div
            key={t.id}
            role={t.tone === 'error' ? 'alert' : 'status'}
            className={`pointer-events-auto flex animate-toast-in items-start gap-2.5 rounded-lg border bg-slate-900/95 px-3 py-2.5 shadow-xl backdrop-blur ${className}`}
          >
            <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-slate-100">{t.title}</p>
              {t.message && <p className="mt-0.5 text-xs text-slate-400">{t.message}</p>}
            </div>
            <button
              type="button"
              onClick={() => onDismiss(t.id)}
              className="rounded p-0.5 text-slate-500 hover:text-slate-200"
              aria-label="Dismiss notification"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── Top bar

function NodePill({ label, detail, health, Icon }) {
  const state = health === undefined ? 'unknown' : health?.online ? 'online' : 'offline';
  const dot = { online: 'bg-emerald-400', offline: 'bg-red-500', unknown: 'bg-slate-500' }[state];
  const text = { online: 'Online', offline: 'Offline', unknown: 'Checking…' }[state];
  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/70 px-2.5 py-1.5"
      title={[health?.detail ?? detail, health?.latency_ms != null ? `${health.latency_ms} ms` : null].filter(Boolean).join(' · ')}
    >
      <Icon className="h-3.5 w-3.5 text-slate-400" aria-hidden />
      <div className="leading-tight">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
        <p className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
          <span className="relative flex h-2 w-2">
            {state === 'online' && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${dot}`} />}
            <span className={`relative inline-flex h-2 w-2 rounded-full ${dot}`} />
          </span>
          {text}
          {state === 'online' && health?.latency_ms != null && (
            <span className="tabular-nums text-slate-500">{health.latency_ms}ms</span>
          )}
        </p>
      </div>
    </div>
  );
}

function Metric({ label, value, Icon, accent = 'text-slate-100' }) {
  return (
    <div className="flex items-center gap-2 px-3">
      <Icon className="h-4 w-4 text-slate-500" aria-hidden />
      <div className="leading-tight">
        <p className={`text-sm font-semibold tabular-nums ${accent}`}>{value}</p>
        <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      </div>
    </div>
  );
}

function TopBar({ metrics, nodes, orchestratorOnline, openCount, onOpenQueue, onRefresh, refreshing }) {
  // Downstream nodes are only reachable through the orchestrator; if it is down, so are they (from our view).
  const downstream = (health) => health ?? (orchestratorOnline === false ? { online: false, detail: 'Orchestrator unreachable' } : undefined);
  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-800 bg-slate-950/90 px-4 py-2.5 backdrop-blur">
      <div className="flex items-center gap-2.5">
        <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 shadow-lg shadow-blue-900/40">
          <Shield className="h-4 w-4 text-white" aria-hidden />
        </div>
        <div className="leading-tight">
          <h1 className="text-sm font-bold tracking-tight text-slate-50">Drake&amp;DonT</h1>
          <p className="text-[10px] text-slate-500">FinCrime &amp; Insider Risk Intelligence</p>
        </div>
        {API_CONFIG.useMocks && (
          <span className="ml-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-amber-300">
            Demo data
          </span>
        )}
      </div>

      <button
        type="button"
        onClick={onOpenQueue}
        className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-slate-600 hover:bg-slate-800"
      >
        <ListChecks className="h-4 w-4 text-sky-400" aria-hidden />
        Alert Queue
        {openCount > 0 && (
          <span className="rounded-full bg-red-500 px-1.5 text-[10px] font-bold text-white">{openCount}</span>
        )}
      </button>

      <div className="hidden items-center divide-x divide-slate-800 xl:flex">
        <Metric label="Open alerts" value={formatNumber(metrics.open_alerts)} Icon={AlertOctagon} accent="text-red-300" />
        <Metric label="Active cases" value={formatNumber(metrics.active_cases)} Icon={Briefcase} />
        <Metric label="Critical" value={formatNumber(metrics.critical_alerts)} Icon={Activity} accent="text-orange-300" />
        <Metric label="Tx scanned" value={formatNumber(metrics.transactions_scanned)} Icon={Database} />
        <Metric label="Flagged staff" value={formatNumber(metrics.flagged_employees)} Icon={Users} />
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <NodePill
          label="Orchestrator"
          detail={API_CONFIG.baseURL}
          health={orchestratorOnline === null ? undefined : { online: orchestratorOnline, detail: API_CONFIG.baseURL }}
          Icon={Server}
        />
        <NodePill label="ML Node" detail="FastAPI :8001" health={downstream(nodes?.ml)} Icon={Cpu} />
        <NodePill label="LLM Node" detail="Ollama" health={downstream(nodes?.llm)} Icon={BrainCircuit} />
        <NodePill label="Graph DB" detail="Neo4j" health={downstream(nodes?.graph)} Icon={Database} />
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          className="rounded-lg border border-slate-800 p-2 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200 disabled:opacity-50"
          aria-label="Refresh data"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
}

// ─────────────────────────────────────────────────────────────── App

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(true);
  const [alertsError, setAlertsError] = useState(null);
  const [selectedAlertId, setSelectedAlertId] = useState(null);

  const [investigation, setInvestigation] = useState(null);
  const [investigationLoading, setInvestigationLoading] = useState(false);
  const [investigationError, setInvestigationError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  const [selectedEntity, setSelectedEntity] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [orchestratorOnline, setOrchestratorOnline] = useState(null);
  const [reviewers, setReviewers] = useState([]);

  const [pendingAction, setPendingAction] = useState(null);
  const [assigningCaseId, setAssigningCaseId] = useState(null);
  const [queueOpen, setQueueOpen] = useState(false);

  const { toasts, pushToast, dismissToast } = useToasts();
  const alertsAbort = useRef(null);

  // ── Data loading ────────────────────────────────────────────────────────

  const loadAlerts = useCallback(async () => {
    alertsAbort.current?.abort();
    const controller = new AbortController();
    alertsAbort.current = controller;
    setAlertsLoading(true);
    setAlertsError(null);
    try {
      const list = await fetchAlerts({ signal: controller.signal });
      setAlerts(list);
      setSelectedAlertId((prev) => (prev && list.some((a) => a.alert_id === prev) ? prev : list[0]?.alert_id ?? null));
    } catch (err) {
      if (err.isAborted) return;
      setAlertsError(err);
      pushToast({ tone: 'error', title: 'Could not load alerts', message: getErrorMessage(err) });
    } finally {
      if (alertsAbort.current === controller) setAlertsLoading(false);
    }
  }, [pushToast]);

  useEffect(() => {
    loadAlerts();
    return () => alertsAbort.current?.abort();
  }, [loadAlerts]);

  useEffect(() => {
    const controller = new AbortController();
    fetchReviewers({ signal: controller.signal })
      .then(setReviewers)
      .catch((err) => !err.isAborted && console.warn('Reviewer list unavailable:', err.message));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedAlertId) {
      setInvestigation(null);
      return undefined;
    }
    const controller = new AbortController();
    setInvestigationLoading(true);
    setInvestigationError(null);
    setSelectedEntity(null);
    setInvestigation((prev) => (prev?.alert.alert_id === selectedAlertId ? prev : null));

    getInvestigationDetails(selectedAlertId, { signal: controller.signal })
      .then((data) => {
        setInvestigation(data);
        setInvestigationLoading(false);
      })
      .catch((err) => {
        if (err.isAborted) return;
        setInvestigationError(err);
        setInvestigationLoading(false);
      });

    return () => controller.abort();
  }, [selectedAlertId, reloadKey]);

  useEffect(() => {
    let controller;
    const poll = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const status = await fetchSystemStatus({ signal: controller.signal });
        setSystemStatus(status);
        setOrchestratorOnline(true);
      } catch (err) {
        if (err.isAborted) return;
        setOrchestratorOnline(false);
        // Downstream health is unknown when the gateway itself is unreachable.
        setSystemStatus((prev) =>
          prev
            ? { ...prev, nodes: Object.fromEntries(Object.entries(prev.nodes).map(([k, v]) => [k, { ...v, online: false }])) }
            : prev,
        );
      }
    };
    poll();
    const timer = setInterval(poll, STATUS_POLL_MS);
    return () => {
      clearInterval(timer);
      controller?.abort();
    };
  }, []);

  // ── Derived state ───────────────────────────────────────────────────────

  const reviewerName = useCallback(
    (id) => (id ? reviewers.find((r) => r.reviewer_id === id)?.name ?? id : 'Unassigned'),
    [reviewers],
  );

  const metrics = useMemo(
    () =>
      systemStatus?.metrics ?? {
        open_alerts: alerts.filter((a) => a.status === 'OPEN').length,
        active_cases: alerts.filter((a) => a.status !== 'CLOSED').length,
        critical_alerts: alerts.filter((a) => a.risk_level === 'CRITICAL').length,
        transactions_scanned: null,
        flagged_employees: new Set(alerts.map((a) => a.emp_id).filter(Boolean)).size,
      },
    [systemStatus, alerts],
  );

  const openCount = alerts.filter((a) => a.status === 'OPEN').length;
  const caseId = investigation?.alert.case_id ?? null;

  // ── Case mutations ──────────────────────────────────────────────────────

  /** Apply a partial alert update everywhere the alert is displayed. */
  const patchCase = useCallback((targetCaseId, patch) => {
    setAlerts((prev) => prev.map((a) => (a.case_id === targetCaseId ? { ...a, ...patch } : a)));
    setInvestigation((prev) =>
      prev && prev.alert.case_id === targetCaseId ? { ...prev, alert: { ...prev.alert, ...patch } } : prev,
    );
  }, []);

  const handleAssign = useCallback(
    async (targetCaseId, reviewerId) => {
      setAssigningCaseId(targetCaseId);
      try {
        const result = await assignCase(targetCaseId, reviewerId);
        patchCase(targetCaseId, { assigned_to: result.assigned_to, status: result.status });
        pushToast({
          tone: 'success',
          title: 'Case assigned',
          message: `${targetCaseId} → ${reviewerName(result.assigned_to)}`,
        });
      } catch (err) {
        pushToast({ tone: 'error', title: 'Assignment failed', message: getErrorMessage(err) });
      } finally {
        setAssigningCaseId(null);
      }
    },
    [patchCase, pushToast, reviewerName],
  );

  const runCaseAction = useCallback(
    async (kind, action, onSuccess, failureTitle) => {
      if (!caseId) return;
      setPendingAction(kind);
      try {
        const result = await action(caseId);
        onSuccess(result);
      } catch (err) {
        pushToast({ tone: 'error', title: failureTitle, message: getErrorMessage(err) });
      } finally {
        setPendingAction(null);
      }
    },
    [caseId, pushToast],
  );

  const handleFreeze = () =>
    runCaseAction(
      'freeze',
      (id) => freezeAccounts(id, investigation.associated_account_ids),
      (res) => {
        patchCase(res.case_id, { status: res.status });
        pushToast({
          tone: 'success',
          title: 'Accounts frozen',
          message: `${res.affected_account_ids.length} accounts on ${res.case_id} are now frozen.`,
        });
      },
      'Freeze failed',
    );

  const handleEscalate = () =>
    runCaseAction(
      'escalate',
      (id) => escalateCase(id, { note: `Escalated from dashboard: ${investigation.alert.title}` }),
      (res) => {
        patchCase(res.case_id, { status: res.status });
        pushToast({ tone: 'success', title: 'Escalated to Compliance Lead', message: res.message ?? res.case_id });
      },
      'Escalation failed',
    );

  const handleExport = () =>
    runCaseAction(
      'export',
      (id) => exportEvidenceDossier(id),
      (res) =>
        pushToast({
          tone: 'success',
          title: 'Evidence packet exported',
          message: `${res.filename} · ${(res.size / 1024).toFixed(1)} KB`,
        }),
      'Export failed',
    );

  const handleSelectAlert = (alertId) => setSelectedAlertId(alertId);

  const handleRefresh = () => {
    loadAlerts();
    setReloadKey((k) => k + 1);
  };

  const handleTimelineSelect = useCallback(
    (event) => setSelectedEntity({ kind: 'event', id: event.event_id, ...event }),
    [],
  );

  // ── Render ──────────────────────────────────────────────────────────────

  const showGraphLoading = investigationLoading && !investigation;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-slate-200">
      <TopBar
        metrics={metrics}
        nodes={systemStatus?.nodes}
        orchestratorOnline={orchestratorOnline}
        openCount={openCount}
        onOpenQueue={() => setQueueOpen(true)}
        onRefresh={handleRefresh}
        refreshing={alertsLoading}
      />

      <main className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3 lg:flex-row lg:overflow-hidden">
        <section
          className="flex min-w-0 shrink-0 flex-col gap-3 lg:min-h-0 lg:shrink lg:basis-3/5"
          aria-label="Graph and timeline"
        >
          <div className="h-[420px] lg:h-auto lg:min-h-[300px] lg:flex-[3]">
            <GraphCanvas
              elements={investigation?.graph ?? null}
              loading={showGraphLoading}
              selectedId={selectedEntity?.id ?? null}
              onSelect={setSelectedEntity}
              title={investigation ? `Entity Link Graph · ${investigation.alert.alert_id}` : 'Entity Link Graph'}
            />
          </div>
          <div className="h-[460px] lg:h-auto lg:min-h-[220px] lg:flex-[2]">
            <TimelineStream
              events={investigation?.timeline ?? []}
              loading={showGraphLoading}
              fraudWindow={investigation?.fraud_window ?? null}
              selectedEventId={selectedEntity?.kind === 'event' ? selectedEntity.event_id : null}
              onEventSelect={handleTimelineSelect}
            />
          </div>
        </section>

        <aside className="h-[720px] min-w-0 shrink-0 lg:h-auto lg:min-h-0 lg:shrink lg:basis-2/5" aria-label="Evidence">
          <EvidencePanel
            investigation={investigation}
            loading={investigationLoading || (alertsLoading && !investigation && !alertsError)}
            error={investigationError ?? (!investigation ? alertsError : null)}
            onRetry={investigationError ? () => setReloadKey((k) => k + 1) : loadAlerts}
            selectedEntity={selectedEntity}
            onClearSelection={() => setSelectedEntity(null)}
            reviewerName={reviewerName}
            pendingAction={pendingAction}
            onFreeze={handleFreeze}
            onEscalate={handleEscalate}
            onExport={handleExport}
          />
        </aside>
      </main>

      <CaseManager
        open={queueOpen}
        onClose={() => setQueueOpen(false)}
        alerts={alerts}
        loading={alertsLoading}
        error={alertsError}
        onRetry={loadAlerts}
        selectedAlertId={selectedAlertId}
        onSelectAlert={handleSelectAlert}
        reviewers={reviewers}
        reviewerName={reviewerName}
        onAssign={handleAssign}
        assigningCaseId={assigningCaseId}
      />

      <Toaster toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
