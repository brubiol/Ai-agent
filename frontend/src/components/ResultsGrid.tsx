import { ResultCard } from './ResultCard';
import type { RephraserState, WritingStyle } from '../types';

interface ResultsGridProps {
  styles: WritingStyle[];
  results: Record<WritingStyle, { text: string; done: boolean }>;
  state: RephraserState;
}

export function ResultsGrid({ styles, results, state }: ResultsGridProps) {
  return (
    <section className="panel results-panel" aria-labelledby="rephrase-results-heading">
      <div className="panel-header">
        <h2 id="rephrase-results-heading">Results</h2>
      </div>
      <div className="results-grid">
        {styles.map((style) => (
          <ResultCard key={style} style={style} content={results[style]?.text ?? ''} done={results[style]?.done ?? false} state={state} />
        ))}
      </div>
    </section>
  );
}

