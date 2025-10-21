import type { WritingStyle } from '../types';

const DEFAULT_BASE_URL = '';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? DEFAULT_BASE_URL;

export interface RephrasePayload {
  text: string;
  styles: WritingStyle[];
}

export interface RephraseResult {
  results: Record<WritingStyle, string>;
}

const jsonHeaders = { 'Content-Type': 'application/json' };

type SSEEventHandler = (event: { event: string; data: Record<string, unknown> }) => void;

function parseSSE(buffer: string, onEvent: SSEEventHandler): string {
  let remaining = buffer;
  let boundary = remaining.indexOf('\n\n');

  while (boundary !== -1) {
    const rawEvent = remaining.slice(0, boundary);
    remaining = remaining.slice(boundary + 2);

    const lines = rawEvent.split('\n');
    let eventType = 'message';
    const payloadLines: string[] = [];

    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        payloadLines.push(line.slice(5).trim());
      }
    }

    const payload = payloadLines.join('\n');
    if (!payload) {
      boundary = remaining.indexOf('\n\n');
      continue;
    }

    try {
      const data = JSON.parse(payload) as Record<string, unknown>;
      onEvent({ event: eventType, data });
    } catch {
      // Ignore malformed payloads to keep the stream flowing.
    }
    boundary = remaining.indexOf('\n\n');
  }

  return remaining;
}

export async function requestRephrase(payload: RephrasePayload, signal?: AbortSignal): Promise<RephraseResult> {
  const response = await fetch(`${API_BASE_URL}/rephrase`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const detail = await safeParseError(response);
    throw new Error(detail ?? `Request failed with status ${response.status}`);
  }

  return (await response.json()) as RephraseResult;
}

async function safeParseError(response: Response): Promise<string | null> {
  try {
    const data = await response.json();
    if (data && typeof data === 'object' && 'detail' in data) {
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }
  } catch {
    // ignore JSON parse failures
  }
  return null;
}

export interface StreamHandlers {
  onChunk: (data: { style?: WritingStyle; delta?: string; done?: boolean }) => void;
  onDone: () => void;
}

export async function requestRephraseStream(
  payload: RephrasePayload,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/rephrase/stream`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await safeParseError(response);
    throw new Error(detail ?? `Streaming request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const abortHandler = () => {
    reader.cancel().catch(() => {
      /* swallow cancellation errors */
    });
  };
  signal?.addEventListener('abort', abortHandler);

  const forwardEvent = ({ event, data }: { event: string; data: Record<string, unknown> }) => {
    if (event === 'chunk') {
      handlers.onChunk({
        style: data.style as WritingStyle | undefined,
        delta: typeof data.delta === 'string' ? data.delta : undefined,
        done: typeof data.done === 'boolean' ? data.done : undefined,
      });
    } else if (event === 'done') {
      handlers.onChunk({ done: true });
    }
  };

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSSE(buffer, forwardEvent);
    }
    buffer += decoder.decode();
    parseSSE(buffer, forwardEvent);
  } finally {
    signal?.removeEventListener('abort', abortHandler);
    handlers.onDone();
    reader.releaseLock();
    try {
      await response.body.cancel();
    } catch {
      // Ignore cancellation failures.
    }
  }
}
