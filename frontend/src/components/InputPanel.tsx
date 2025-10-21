import type { ChangeEvent } from 'react';

import { STYLE_LABELS, WRITING_STYLES, type WritingStyle } from '../types';

interface InputPanelProps {
  text: string;
  onTextChange: (value: string) => void;
  selectedStyles: WritingStyle[];
  onToggleStyle: (style: WritingStyle) => void;
  disabled?: boolean;
}

export function InputPanel({ text, onTextChange, selectedStyles, onToggleStyle, disabled = false }: InputPanelProps) {
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
        <span className="muted">{text.length} characters</span>
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

