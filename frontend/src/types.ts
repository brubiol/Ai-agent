export type WritingStyle = 'professional' | 'casual' | 'polite' | 'social';

export const WRITING_STYLES: WritingStyle[] = ['professional', 'casual', 'polite', 'social'];

export const STYLE_LABELS: Record<WritingStyle, string> = {
  professional: 'Professional',
  casual: 'Casual',
  polite: 'Polite',
  social: 'Social',
};

export type RephraserState = 'idle' | 'processing' | 'streaming' | 'canceled' | 'done' | 'error';

