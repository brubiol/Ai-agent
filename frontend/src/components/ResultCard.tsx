import { useEffect, useState } from 'react';

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
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(id);
  }, [copied]);

  const handleCopy = async () => {
    if (!content) return;
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className="result-card" aria-live="polite">
      <header>
        <h3>{STYLE_LABELS[style]}</h3>
        <div className="result-card-actions">
          <StatusPill isStreaming={isStreaming} done={done} hasContent={content.length > 0} />
          <button type="button" className={`copy-button${copied ? ' copied' : ''}`} onClick={handleCopy} disabled={!content}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </header>
      <div className="result-content">
        {content ? (
          <p>{content}</p>
        ) : isPending ? (
          <Skeleton />
        ) : (
          <p className="muted">No result yet.</p>
        )}
      </div>
    </article>
  );
}

function Skeleton() {
  return (
    <div className="result-skeleton" aria-hidden="true">
      <span className="skeleton-line skeleton-line-lg" />
      <span className="skeleton-line" />
      <span className="skeleton-line" />
      <span className="skeleton-line" />
    </div>
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
  if (hasContent) {
    return <span className="pill pill-idle">Ready</span>;
  }
  return <span className="pill pill-idle">Idle</span>;
}
