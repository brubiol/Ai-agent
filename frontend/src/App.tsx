import { useMemo, useState } from 'react';

import { Controls } from './components/Controls';
import { InputPanel } from './components/InputPanel';
import { ResultsGrid } from './components/ResultsGrid';
import { useRephraser } from './hooks/useRephraser';
import { STYLE_LABELS, WRITING_STYLES, type WritingStyle } from './types';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [selectedStyles, setSelectedStyles] = useState<WritingStyle[]>([...WRITING_STYLES]);
  const { state, error, results, process, stream, cancel } = useRephraser();

  const canSubmit = text.trim().length > 0 && selectedStyles.length > 0;

  const summary = useMemo(() => {
    if (state === 'processing' || state === 'streaming') {
      return `Rephrasing into ${selectedStyles.length} style${selectedStyles.length > 1 ? 's' : ''}...`;
    }
    if (state === 'done') {
      return `Completed: ${selectedStyles.map((style) => STYLE_LABELS[style]).join(', ')}`;
    }
    if (state === 'canceled') {
      return 'Request canceled.';
    }
    if (state === 'error' && error) {
      return `Error: ${error}`;
    }
    return 'Provide text and choose the styles to rephrase.';
  }, [state, selectedStyles, error]);

  const handleToggleStyle = (style: WritingStyle) => {
    setSelectedStyles((prev) => {
      if (prev.includes(style)) {
        return prev.filter((item) => item !== style);
      }
      return [...prev, style];
    });
  };

  const handleProcess = () => {
    if (!canSubmit) return;
    void process(text, selectedStyles);
  };

  const handleStream = () => {
    if (!canSubmit) return;
    void stream(text, selectedStyles);
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Rephrasing Workbench</h1>
        <p className="muted">{summary}</p>
      </header>
      <InputPanel text={text} onTextChange={setText} selectedStyles={selectedStyles} onToggleStyle={handleToggleStyle} disabled={state === 'processing' || state === 'streaming'} />
      <Controls state={state} onProcess={handleProcess} onStream={handleStream} onCancel={cancel} canSubmit={canSubmit} />
      <ResultsGrid styles={WRITING_STYLES} results={results} state={state} />
      {error && state === 'error' && (
        <div role="alert" className="error-banner">
          {error}
        </div>
      )}
    </main>
  );
}

export default App;
