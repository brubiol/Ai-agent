import type { ChangeEvent } from 'react';

import { STYLE_LABELS, WRITING_STYLES, type WritingStyle } from '../types';

interface InputPanelProps {
  text: string;
  onTextChange: (value: string) => void;
  selectedStyles: WritingStyle[];
  onToggleStyle: (style: WritingStyle) => void;
  disabled?: boolean;
  charCount: number;
  wordCount: number;
  maxChars: number;
}

export function InputPanel({
  text,
  onTextChange,
  selectedStyles,
  onToggleStyle,
  disabled = false,
  charCount,
  wordCount,
  maxChars,
}: InputPanelProps) {
  const handleTextChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onTextChange(event.target.value);
  };

  const handleStyleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onToggleStyle(event.target.value as WritingStyle);
  };

  return (
    <section className="panel input-panel" aria-labelledby="rephrase-input-heading">
      <div className="panel-header">
        <h2 id="rephrase-input-heading">Content</h2>
        <span className={`muted counter${charCount > maxChars ? ' counter-alert' : ''}`}>
          {wordCount} word{wordCount === 1 ? '' : 's'} · {charCount}/{maxChars} characters
        </span>
      </div>
      <label className="sr-only" htmlFor="rephrase-text">
        Text to rephrase
      </label>
      <textarea
        id="rephrase-text"
        value={text}
        onChange={handleTextChange}
        disabled={disabled}
        placeholder="Paste or type the message you want to rephrase..."
        rows={8}
      />
      {charCount > maxChars ? (
        <p className="input-warning" role="alert">
          Trim the text by {charCount - maxChars} character{charCount - maxChars === 1 ? '' : 's'} to continue.
        </p>
      ) : null}

      <fieldset className="style-selector">
        <legend>Styles</legend>
        <div className="style-grid">
          {WRITING_STYLES.map((style) => (
            <label key={style} className="style-option">
              <input
                type="checkbox"
                value={style}
                onChange={handleStyleChange}
                checked={selectedStyles.includes(style)}
                disabled={disabled}
              />
              <span>{STYLE_LABELS[style]}</span>
            </label>
          ))}
        </div>
      </fieldset>
    </section>
  );
}
