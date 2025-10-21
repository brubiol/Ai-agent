import { STYLE_LABELS, type RephraserState, type WritingStyle } from '../types';

interface ResultCardProps {
  style: WritingStyle;
  content: string;
  done: boolean;
  state: RephraserState;
}

export function ResultCard({ style, content, done, state }: ResultCardProps) {
  const isPending = (state === 'processing' || state === 'streaming') && !done && !content;
  const isStreaming = state === 'streaming' && !done;

  return (
    <article className="result-card" aria-live="polite">
      <header>
        <h3>{STYLE_LABELS[style]}</h3>
        <StatusPill isStreaming={isStreaming} done={done} hasContent={content.length > 0} />
      </header>
      <div className="result-content">
        {content ? (
          <p>{content}</p>
        ) : isPending ? (
          <p className="muted">Awaiting output...</p>
        ) : (
          <p className="muted">No result yet.</p>
        )}
      </div>
    </article>
  );
}

function StatusPill({ isStreaming, done, hasContent }: { isStreaming: boolean; done: boolean; hasContent: boolean }) {
  if (isStreaming) {
    return (
      <span className="pill pill-streaming">
        <span className="spinner spinner-inline" aria-hidden="true" />
        <span>Streaming</span>
      </span>
    );
  }
  if (done && hasContent) {
    return <span className="pill pill-done">Done</span>;
  }
  return <span className="pill pill-idle">Idle</span>;
}

