import { useEffect, useMemo, useState } from 'react';

import { Controls } from './components/Controls';
import { InputPanel } from './components/InputPanel';
import { ResultsGrid } from './components/ResultsGrid';
import { Toast } from './components/Toast';
import { useRephraser } from './hooks/useRephraser';
import { STYLE_LABELS, WRITING_STYLES, type WritingStyle } from './types';
import './App.css';

const STORAGE_KEY = 'rephraser:last-state';
const MAX_TEXT_LENGTH = 2000;
const ALLOWED_STYLE_SET = new Set<WritingStyle>(WRITING_STYLES);

function App() {
  const [text, setText] = useState('');
  const [selectedStyles, setSelectedStyles] = useState<WritingStyle[]>([...WRITING_STYLES]);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const { state, error, results, process, stream, cancel } = useRephraser();

  const charCount = text.length;
  const wordCount = text.trim().length ? text.trim().split(/\s+/).length : 0;
  const isOverLimit = charCount > MAX_TEXT_LENGTH;
  const canSubmit = text.trim().length > 0 && selectedStyles.length > 0 && !isOverLimit;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (!saved) return;
      const parsed: Partial<{ text: string; styles: WritingStyle[] }> = JSON.parse(saved);
      if (typeof parsed.text === 'string') {
        setText(parsed.text.slice(0, MAX_TEXT_LENGTH));
      }
      if (Array.isArray(parsed.styles)) {
        const restored = parsed.styles.filter(
          (style): style is WritingStyle => typeof style === 'string' && ALLOWED_STYLE_SET.has(style as WritingStyle),
        );
        if (restored.length > 0) {
          setSelectedStyles(restored);
        }
      }
    } catch {
      // Ignore malformed saved state.
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const payload = JSON.stringify({ text, styles: selectedStyles });
    window.localStorage.setItem(STORAGE_KEY, payload);
  }, [text, selectedStyles]);

  useEffect(() => {
    if (state === 'error' && error) {
      setToastMessage(error);
    }
  }, [state, error]);

  const dismissToast = () => setToastMessage(null);

  const summary = useMemo(() => {
    if (isOverLimit) {
      return `Text is too long by ${charCount - MAX_TEXT_LENGTH} character${charCount - MAX_TEXT_LENGTH === 1 ? '' : 's'}.`;
    }
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
  }, [state, selectedStyles, error, isOverLimit, charCount]);

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
      <InputPanel
        text={text}
        onTextChange={setText}
        selectedStyles={selectedStyles}
        onToggleStyle={handleToggleStyle}
        disabled={state === 'processing' || state === 'streaming'}
        charCount={charCount}
        wordCount={wordCount}
        maxChars={MAX_TEXT_LENGTH}
      />
      <Controls state={state} onProcess={handleProcess} onStream={handleStream} onCancel={cancel} canSubmit={canSubmit} />
      <ResultsGrid styles={WRITING_STYLES} results={results} state={state} />
      {toastMessage ? (
        <div className="toast-container">
          <Toast message={toastMessage} onDismiss={dismissToast} />
        </div>
      ) : null}
    </main>
  );
}

export default App;
