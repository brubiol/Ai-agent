import type { RephraserState } from '../types';

interface ControlsProps {
  state: RephraserState;
  onProcess: () => void;
  onStream: () => void;
  onCancel: () => void;
  canSubmit: boolean;
}

const busyStates: RephraserState[] = ['processing', 'streaming'];

export function Controls({ state, onProcess, onStream, onCancel, canSubmit }: ControlsProps) {
  const isBusy = busyStates.includes(state);

  return (
    <section className="panel controls-panel" aria-label="Actions">
      <div className="controls">
        <button type="button" onClick={onProcess} disabled={!canSubmit || isBusy}>
          Process
        </button>
        <button type="button" onClick={onStream} disabled={!canSubmit || isBusy}>
          Stream
        </button>
        <button type="button" onClick={onCancel} disabled={!isBusy}>
          Cancel
        </button>
      </div>
      <StatusBadge state={state} />
    </section>
  );
}

function StatusBadge({ state }: { state: RephraserState }) {
  const message = (() => {
    switch (state) {
      case 'processing':
        return 'Processing...';
      case 'streaming':
        return 'Streaming...';
      case 'canceled':
        return 'Canceled';
      case 'done':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Idle';
    }
  })();

  return (
    <div className={`status-badge status-${state}`}>
      {(state === 'processing' || state === 'streaming') && <span className="spinner" aria-hidden="true" />}
      <span>{message}</span>
    </div>
  );
}
