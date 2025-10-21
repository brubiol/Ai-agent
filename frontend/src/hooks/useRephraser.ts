import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { requestRephrase, requestRephraseStream } from '../lib/api';
import { WRITING_STYLES, type RephraserState, type WritingStyle } from '../types';

type StyleResult = {
  text: string;
  done: boolean;
};

type ResultsMap = Record<WritingStyle, StyleResult>;

const createInitialResults = (): ResultsMap =>
  WRITING_STYLES.reduce<ResultsMap>((acc, style) => {
    acc[style] = { text: '', done: false };
    return acc;
  }, {} as ResultsMap);

export interface UseRephraserOptions {
  styles?: WritingStyle[];
}

export function useRephraser(initialStyles: WritingStyle[] = WRITING_STYLES) {
  const [state, setState] = useState<RephraserState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultsMap>(createInitialResults);
  const styles = useMemo(() => initialStyles, [initialStyles]);

  const controllerRef = useRef<AbortController | null>(null);
  const canceledRef = useRef(false);
  const activeStylesRef = useRef<Set<WritingStyle>>(new Set());
  const ignoreUpdatesRef = useRef(false);

  const resetResults = useCallback(
    (requestStyles: WritingStyle[]) => {
      ignoreUpdatesRef.current = false;
      setResults((prev) => {
        const next: ResultsMap = { ...prev };
        for (const style of WRITING_STYLES) {
          if (requestStyles.includes(style)) {
            next[style] = { text: '', done: false };
          } else {
            const previous = prev[style] ?? { text: '', done: true };
            next[style] = { text: previous.text, done: true };
          }
        }
        return next;
      });
    },
    [],
  );

  const clearController = useCallback(() => {
    const wasCanceled = canceledRef.current;
    controllerRef.current = null;
    canceledRef.current = false;
    activeStylesRef.current.clear();
    if (wasCanceled) {
      setState((current) => (current === 'canceled' ? 'idle' : current));
    }
  }, []);

  const process = useCallback(
    async (text: string, requestStyles: WritingStyle[]) => {
      controllerRef.current?.abort();
      canceledRef.current = false;

      const controller = new AbortController();
      controllerRef.current = controller;

      setError(null);
      setState('processing');
      resetResults(requestStyles);
      activeStylesRef.current = new Set(requestStyles);

      try {
        const { results: data } = await requestRephrase({ text, styles: requestStyles }, controller.signal);
      ignoreUpdatesRef.current = false;
      setResults((prev) => {
          const next = createInitialResults();
          for (const style of WRITING_STYLES) {
            const textResult = data?.[style] ?? '';
            next[style] = {
              text: requestStyles.includes(style) ? textResult : prev[style]?.text ?? '',
              done: requestStyles.includes(style) ? true : prev[style]?.done ?? false,
            };
          }
          return next;
        });
        if (!controller.signal.aborted) {
          setState('done');
        }
      } catch (err) {
        if (controller.signal.aborted || canceledRef.current) {
          setState('canceled');
        } else {
          setError(err instanceof Error ? err.message : 'Unexpected error');
          setState('error');
        }
      } finally {
        clearController();
      }
    },
    [resetResults, clearController],
  );

  const stream = useCallback(
    async (text: string, requestStyles: WritingStyle[]) => {
      controllerRef.current?.abort();
      canceledRef.current = false;

      const controller = new AbortController();
      controllerRef.current = controller;

      setError(null);
      setState('streaming');
      resetResults(requestStyles);
      activeStylesRef.current = new Set(requestStyles);

      try {
        await requestRephraseStream(
          { text, styles: requestStyles },
          {
            onChunk: ({ style, delta, done }) => {
              if (style) {
                setResults((prev) => {
                  if (ignoreUpdatesRef.current) {
                    return prev;
                  }
                  const current = prev[style] ?? { text: '', done: false };
                  const nextDelta = delta ?? '';
                  const needsSpace =
                    current.text.length > 0 &&
                    nextDelta.length > 0 &&
                    !/^\s/.test(nextDelta) &&
                    !/[ \n]$/.test(current.text);
                  return {
                    ...prev,
                    [style]: {
                      text: current.text + (needsSpace ? ` ${nextDelta}` : nextDelta),
                      done: done ?? current.done,
                    },
                  };
                });
                if (done) {
                  activeStylesRef.current.delete(style);
                }
              }
              if (done && !style) {
                setState((existing) => (existing === 'canceled' ? existing : 'done'));
              }
            },
            onDone: () => {
              if (!controller.signal.aborted && !canceledRef.current) {
                setState(activeStylesRef.current.size === 0 ? 'done' : 'streaming');
              }
            },
          },
          controller.signal,
        );
      } catch (err) {
        if (controller.signal.aborted || canceledRef.current) {
          setState('canceled');
        } else {
          setError(err instanceof Error ? err.message : 'Unexpected error');
          setState('error');
        }
      } finally {
        clearController();
      }
    },
    [resetResults, clearController],
  );

  const cancel = useCallback(() => {
    if (controllerRef.current) {
      canceledRef.current = true;
       ignoreUpdatesRef.current = true;
      controllerRef.current.abort();
      setState('canceled');
    }
  }, []);

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  const isBusy = state === 'processing' || state === 'streaming';

  return {
    state,
    error,
    results,
    styles,
    isBusy,
    process,
    stream,
    cancel,
  };
}
