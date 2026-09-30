/**
 * @file Evidence panel: composite risk, dual signal gauges, LLM explanation,
 * regulatory breach tags, step-by-step audit rationale and case workflow actions.
 */

import { useEffect, useState } from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  ArrowUpRight,
  BrainCircuit,
  ClipboardList,
  Download,
  FileSearch,
  Gavel,
  Info,
  Loader2,
  MousePointerClick,
  Network,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Snowflake,
  Sparkles,
  UserRound,
  X,
} from 'lucide-react';
import { formatCurrency, formatDate, formatDateTime, formatPercent, humanize } from '../utils/format';

const RISK_STYLES = {
  CRITICAL: { ring: 'bg-red-500/15 text-red-300 ring-red-500/50', bar: 'bg-red-500', Icon: AlertOctagon },
  HIGH: { ring: 'bg-orange-500/15 text-orange-300 ring-orange-500/50', bar: 'bg-orange-500', Icon: ShieldAlert },
  MEDIUM: { ring: 'bg-amber-500/15 text-amber-300 ring-amber-500/50', bar: 'bg-amber-400', Icon: AlertTriangle },
  LOW: { ring: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/50', bar: 'bg-emerald-500', Icon: ShieldCheck },
};

const STATUS_STYLES = {
  OPEN: 'bg-sky-500/15 text-sky-300',
  ASSIGNED: 'bg-indigo-500/15 text-indigo-300',
  ESCALATED: 'bg-amber-500/15 text-amber-300',
  FROZEN: 'bg-cyan-500/15 text-cyan-200',
  CLOSED: 'bg-slate-700/60 text-slate-300',
};

function breachTagClass(tag) {
  const t = tag.toLowerCase();
  if (t.includes('bsa') || t.includes('structuring') || t.includes('aml')) return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
  if (t.includes('insider') || t.includes('unauthori') || t.includes('tamper')) return 'border-red-500/40 bg-red-500/10 text-red-200';
  return 'border-slate-600 bg-slate-800/60 text-slate-300';
}

export function RiskBadge({ level, size = 'md' }) {
  const style = RISK_STYLES[level] ?? RISK_STYLES.MEDIUM;
  const { Icon } = style;
  const sizing = size === 'sm' ? 'px-1.5 py-0.5 text-[10px] gap-1' : 'px-2.5 py-1 text-xs gap-1.5';
  return (
    <span className={`inline-flex items-center rounded-md font-bold tracking-wider ring-1 ring-inset ${style.ring} ${sizing}`}>
      <Icon className={size === 'sm' ? 'h-3 w-3' : 'h-3.5 w-3.5'} aria-hidden />
      {level}
    </span>
  );
}

/** Semicircular gauge for a 0..1 score. */
function Gauge({ value, label, caption, color, Icon }) {
  const pct = Math.round(Math.min(1, Math.max(0, value ?? 0)) * 100);
  const radius = 52;
  const length = Math.PI * radius;
  const arc = 'M 8 62 A 52 52 0 0 1 112 62';
  return (
    <div className="flex flex-col items-center rounded-xl border border-slate-800 bg-slate-950/50 px-3 pb-3 pt-2">
      <svg viewBox="0 0 120 70" className="w-full max-w-[150px]" role="img" aria-label={`${label}: ${pct} percent`}>
        <path d={arc} fill="none" stroke="#E2E8F0" strokeWidth="10" strokeLinecap="round" />
        <path
          d={arc}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={length}
          strokeDashoffset={length * (1 - pct / 100)}
          style={{ transition: 'stroke-dashoffset 700ms ease' }}
        />
        <text x="60" y="56" textAnchor="middle" fill="#0F172A" fontSize="22" fontWeight="700">
          {pct}
        </text>
        <text x="60" y="68" textAnchor="middle" fill="#64748B" fontSize="8">
          / 100
        </text>
      </svg>
      <div className="mt-1 flex items-center gap-1.5 text-xs font-semibold text-slate-200">
        <Icon className="h-3.5 w-3.5" style={{ color }} aria-hidden />
        {label}
      </div>
      <p className="mt-0.5 text-center text-[10px] text-slate-500">{caption}</p>
    </div>
  );
}

function Section({ title, Icon, children, right }) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="panel-title flex items-center gap-2">
          <Icon className="h-3.5 w-3.5 text-slate-500" aria-hidden />
          {title}
        </h3>
        {right}
      </div>
      {children}
    </section>
  );
}

const ENTITY_FIELD_ORDER = [
  'emp_id', 'name', 'department', 'role', 'access_tier',
  'account_id', 'customer_name', 'status', 'balance', 'creation_date',
  'tx_id', 'sender_account_id', 'receiver_account_id', 'amount', 'channel', 'timestamp',
];

function formatField(key, value) {
  if (value === null || value === undefined || value === '') return '—';
  if (key === 'balance' || key === 'amount') return formatCurrency(Number(value));
  if (key === 'timestamp') return formatDateTime(value);
  if (key === 'creation_date') return formatDate(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** Inspector for whatever the analyst clicked in the graph or timeline. */
function SelectedEntityCard({ entity, onClear }) {
  const kindLabel =
    entity.kind === 'event' ? `Timeline · ${humanize(entity.category)}` : entity.kind === 'edge' ? `Edge · ${entity.type}` : entity.type;

  const fields =
    entity.kind === 'node'
      ? Object.entries(entity.data ?? {})
      : entity.kind === 'edge'
        ? [
            ['source', entity.source],
            ['target', entity.target],
            ['action_type', entity.action_type],
            ['amount', entity.amount],
            ['timestamp', entity.timestamp],
          ].filter(([, v]) => v !== null && v !== undefined)
        : [
            ['timestamp', entity.timestamp],
            ['emp_id', entity.emp_id],
            ['account_id', entity.account_id],
            ['counterparty', entity.counterparty_account_id],
            ['amount', entity.amount],
            ['channel', entity.channel],
          ].filter(([, v]) => v !== null && v !== undefined);

  const sorted = [...fields].sort(
    ([a], [b]) => (ENTITY_FIELD_ORDER.indexOf(a) + 1 || 99) - (ENTITY_FIELD_ORDER.indexOf(b) + 1 || 99),
  );

  return (
    <section className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-4">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="panel-title flex items-center gap-1.5 text-sky-300">
            <MousePointerClick className="h-3.5 w-3.5" aria-hidden />
            Selected · {kindLabel}
          </p>
          <p className="mt-1 truncate font-mono text-sm text-slate-100">{entity.label ?? entity.title ?? entity.id}</p>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          aria-label="Clear selection"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      {entity.anomalous && (
        <p className="mb-2 inline-flex items-center gap-1 rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-red-300">
          <AlertTriangle className="h-3 w-3" aria-hidden /> Flagged anomalous
        </p>
      )}
      {entity.risk_reasons?.length > 0 && (
        <ul className="mb-2 space-y-0.5 text-xs text-red-200/90">
          {entity.risk_reasons.map((r) => (
            <li key={r}>• {r}</li>
          ))}
        </ul>
      )}
      {entity.kind === 'event' && entity.description && <p className="mb-2 text-xs text-slate-400">{entity.description}</p>}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        {sorted.map(([key, value]) => (
          <div key={key} className="min-w-0">
            <dt className="text-[10px] uppercase tracking-wider text-slate-500">{humanize(key)}</dt>
            <dd className="truncate font-mono text-slate-200" title={String(value)}>
              {formatField(key, value)}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ActionButton({ onClick, disabled, busy, tone, Icon, children }) {
  const tones = {
    sky: 'border-sky-500/40 bg-sky-500/10 text-sky-200 hover:bg-sky-500/20',
    skyConfirm: 'border-sky-400 bg-sky-500 text-slate-950 hover:bg-sky-400',
    amber: 'border-amber-500/40 bg-amber-500/10 text-amber-200 hover:bg-amber-500/20',
    emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20',
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      className={`flex min-h-[40px] flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${tones[tone]}`}
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Icon className="h-4 w-4" aria-hidden />}
      {children}
    </button>
  );
}

function PanelSkeleton() {
  return (
    <div className="space-y-4 p-4" role="status" aria-label="Loading evidence">
      <div className="skeleton h-6 w-3/4" />
      <div className="skeleton h-4 w-1/2" />
      <div className="grid grid-cols-2 gap-3">
        <div className="skeleton h-36" />
        <div className="skeleton h-36" />
      </div>
      <div className="skeleton h-32" />
      <div className="skeleton h-8 w-2/3" />
      <div className="space-y-2">
        <div className="skeleton h-12" />
        <div className="skeleton h-12" />
        <div className="skeleton h-12" />
      </div>
    </div>
  );
}

function CenteredState({ Icon, title, body, action, tone = 'slate' }) {
  return (
    <div className="flex h-full flex-col items-center justify-center p-8 text-center">
      <div
        className={`mb-3 grid h-12 w-12 place-items-center rounded-full border ${
          tone === 'red' ? 'border-red-500/30 bg-red-500/10 text-red-400' : 'border-slate-800 bg-slate-900 text-slate-500'
        }`}
      >
        <Icon className="h-5 w-5" aria-hidden />
      </div>
      <p className="text-sm font-medium text-slate-200">{title}</p>
      <p className="mt-1 max-w-xs text-xs text-slate-500">{body}</p>
      {action}
    </div>
  );
}

/**
 * @param {{
 *   investigation: import('../api/client').Investigation | null,
 *   loading?: boolean,
 *   error?: Error | null,
 *   onRetry?: () => void,
 *   selectedEntity?: Record<string, any> | null,
 *   onClearSelection?: () => void,
 *   reviewerName?: (id: string|null) => string,
 *   pendingAction?: 'freeze'|'escalate'|'export'|null,
 *   onFreeze?: () => void,
 *   onEscalate?: () => void,
 *   onExport?: () => void,
 * }} props
 */
export default function EvidencePanel({
  investigation,
  loading = false,
  error = null,
  onRetry,
  selectedEntity = null,
  onClearSelection,
  reviewerName = (id) => id ?? 'Unassigned',
  pendingAction = null,
  onFreeze,
  onEscalate,
  onExport,
}) {
  const [confirmFreeze, setConfirmFreeze] = useState(false);

  // Freeze requires a second click within 4 s.
  useEffect(() => {
    if (!confirmFreeze) return undefined;
    const t = setTimeout(() => setConfirmFreeze(false), 4000);
    return () => clearTimeout(t);
  }, [confirmFreeze]);

  useEffect(() => setConfirmFreeze(false), [investigation?.alert.alert_id]);

  const shell = (content) => (
    <div className="flex h-full min-h-[420px] flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40">
      {content}
    </div>
  );

  if (loading && !investigation) return shell(<PanelSkeleton />);

  if (error && !investigation) {
    return shell(
      <CenteredState
        Icon={AlertTriangle}
        tone="red"
        title="Could not load evidence"
        body={error.message}
        action={
          onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-4 inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden /> Retry
            </button>
          )
        }
      />,
    );
  }

  if (!investigation) {
    return shell(
      <CenteredState
        Icon={FileSearch}
        title="No case selected"
        body="Choose an alert from the queue to review its evidence, AI explanation and recommended actions."
      />,
    );
  }

  const { alert, explanation, associated_account_ids: accountIds } = investigation;
  const risk = RISK_STYLES[alert.risk_level] ?? RISK_STYLES.MEDIUM;
  const divergence = Math.abs(alert.graph_score - alert.ml_score);
  const isClosed = alert.status === 'CLOSED';
  const busy = pendingAction !== null;

  const handleFreeze = () => {
    if (!confirmFreeze) {
      setConfirmFreeze(true);
      return;
    }
    setConfirmFreeze(false);
    onFreeze?.();
  };

  return shell(
    <>
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
              <span className="font-mono">{alert.alert_id}</span>
              <span aria-hidden>·</span>
              <span className="font-mono">{alert.case_id}</span>
              <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_STYLES[alert.status]}`}>
                {alert.status}
              </span>
              {loading && <Loader2 className="h-3 w-3 animate-spin text-sky-400" aria-label="Refreshing" />}
            </div>
            <h2 className="mt-1 text-base font-semibold leading-snug text-slate-50">{alert.title}</h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
              <span className="inline-flex items-center gap-1">
                <UserRound className="h-3.5 w-3.5 text-blue-400" aria-hidden />
                {alert.employee_name ?? 'Unknown'} <span className="font-mono text-slate-500">({alert.emp_id ?? '—'})</span>
              </span>
              <span>Raised {formatDateTime(alert.created_at)}</span>
              <span>
                Reviewer: <span className="text-slate-200">{reviewerName(alert.assigned_to)}</span>
              </span>
            </div>
          </div>
          <RiskBadge level={alert.risk_level} />
        </div>
      </header>

      {/* Body */}
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {selectedEntity && <SelectedEntityCard entity={selectedEntity} onClear={onClearSelection} />}

        {/* Composite risk */}
        <Section
          title="Composite risk"
          Icon={ShieldAlert}
          right={
            <span className="text-xs text-slate-400">
              <span className="text-lg font-bold tabular-nums text-slate-50">{alert.composite_score}</span> / 100
            </span>
          }
        >
          <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full rounded-full transition-all duration-700 ${risk.bar}`}
              style={{ width: `${alert.composite_score}%` }}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Gauge
              value={alert.graph_score}
              label="Graph Pattern Match"
              caption="Neo4j topology: structuring, cycles, fan-out"
              color="#A855F7"
              Icon={Network}
            />
            <Gauge
              value={alert.ml_score}
              label="ML Anomaly Score"
              caption="Behavioural model vs. peer baseline"
              color="#38BDF8"
              Icon={BrainCircuit}
            />
          </div>
          {divergence >= 0.3 && (
            <p className="mt-3 flex items-start gap-1.5 rounded-lg bg-amber-500/10 px-2.5 py-2 text-[11px] text-amber-200">
              <Info className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
              The two signals disagree by {formatPercent(divergence)}. Weigh the underlying evidence before acting.
            </p>
          )}
          {alert.patterns.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {alert.patterns.map((p) => (
                <span key={p} className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                  {p}
                </span>
              ))}
            </div>
          )}
        </Section>

        {/* LLM narrative */}
        <section className="relative overflow-hidden rounded-xl border border-violet-500/30 bg-gradient-to-br from-violet-500/10 via-slate-900/60 to-slate-900/60 p-4">
          <h3 className="panel-title mb-2 flex items-center gap-2 text-violet-300">
            <Sparkles className="h-3.5 w-3.5" aria-hidden />
            Why this alert was generated
          </h3>
          <p className="text-sm leading-relaxed text-slate-200">{explanation.summary}</p>
          <p className="mt-3 flex flex-wrap items-center gap-x-2 text-[10px] text-slate-500">
            {explanation.model && <span>Generated by {explanation.model}</span>}
            {explanation.confidence !== null && <span>· confidence {formatPercent(explanation.confidence)}</span>}
            {explanation.generated_at && <span>· {formatDateTime(explanation.generated_at)}</span>}
            <span>· AI-generated; analyst review required.</span>
          </p>
        </section>

        {/* Breach tags */}
        {explanation.breach_tags.length > 0 && (
          <Section title="Regulatory breach indicators" Icon={Gavel}>
            <div className="flex flex-wrap gap-2">
              {explanation.breach_tags.map((tag) => (
                <span key={tag} className={`rounded-full border px-2.5 py-1 text-xs font-medium ${breachTagClass(tag)}`}>
                  {tag}
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* Rationale */}
        <Section title="Audit rationale" Icon={ClipboardList}>
          {explanation.rationale.length === 0 ? (
            <p className="text-xs text-slate-500">No structured rationale was returned for this alert.</p>
          ) : (
            <ol className="space-y-3">
              {explanation.rationale.map((step) => (
                <li key={step.step} className="flex gap-3">
                  <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-slate-800 text-[11px] font-bold text-slate-200 ring-1 ring-slate-700">
                    {step.step}
                  </span>
                  <div className="min-w-0 pt-0.5">
                    <p className="text-sm font-medium text-slate-100">{step.title}</p>
                    {step.detail && <p className="mt-0.5 text-xs leading-relaxed text-slate-400">{step.detail}</p>}
                    {step.evidence_refs.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {step.evidence_refs.map((ref) => (
                          <span
                            key={ref}
                            className="rounded border border-slate-700 bg-slate-950 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
                          >
                            {ref}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ol>
          )}
        </Section>
      </div>

      {/* Actions */}
      <footer className="border-t border-slate-800 bg-slate-900/90 p-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <ActionButton
            tone={confirmFreeze ? 'skyConfirm' : 'sky'}
            Icon={Snowflake}
            onClick={handleFreeze}
            busy={pendingAction === 'freeze'}
            disabled={busy || isClosed || alert.status === 'FROZEN' || accountIds.length === 0}
          >
            {alert.status === 'FROZEN'
              ? 'Accounts Frozen'
              : confirmFreeze
                ? `Confirm freeze (${accountIds.length})`
                : 'Freeze Associated Accounts'}
          </ActionButton>
          <ActionButton
            tone="amber"
            Icon={ArrowUpRight}
            onClick={onEscalate}
            busy={pendingAction === 'escalate'}
            disabled={busy || isClosed || alert.status === 'ESCALATED'}
          >
            {alert.status === 'ESCALATED' ? 'Escalated' : 'Escalate to Compliance Lead'}
          </ActionButton>
          <ActionButton tone="emerald" Icon={Download} onClick={onExport} busy={pendingAction === 'export'} disabled={busy}>
            Export Evidence Packet
          </ActionButton>
        </div>
        {confirmFreeze && (
          <p className="mt-2 text-center text-[11px] text-sky-300">
            Freezes {accountIds.join(', ')}. Click again to confirm.
          </p>
        )}
      </footer>
    </>,
  );
}
