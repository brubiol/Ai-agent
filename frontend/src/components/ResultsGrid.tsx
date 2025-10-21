import { ResultCard } from './ResultCard';
import type { RephraserState, WritingStyle } from '../types';

interface ResultsGridProps {
  styles: WritingStyle[];
  results: Record<WritingStyle, { text: string; done: boolean }>;
  state: RephraserState;
}

export function ResultsGrid({ styles, results, state }: ResultsGridProps) {
  const hasAnyContent = styles.some((style) => (results[style]?.text ?? '').trim().length > 0);

  return (
    <section className="panel results-panel" aria-labelledby="rephrase-results-heading">
      <div className="panel-header">
        <h2 id="rephrase-results-heading">Results</h2>
      </div>
      {!hasAnyContent && state === 'idle' ? (
        <div className="results-empty" role="status">
          <p className="muted">
            Rewrites will appear here. Choose your styles, paste some text, and press <span className="highlight">Process</span> or{' '}
            <span className="highlight">Stream</span>.
          </p>
        </div>
      ) : null}
      <div className="results-grid">
        {styles.map((style) => (
          <ResultCard key={style} style={style} content={results[style]?.text ?? ''} done={results[style]?.done ?? false} state={state} />
        ))}
      </div>
    </section>
  );
}
