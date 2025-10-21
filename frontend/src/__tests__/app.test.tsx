import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import App from '../App';
import type { StreamHandlers } from '../lib/api';
import { WRITING_STYLES } from '../types';

jest.mock('../lib/api', () => {
  return {
    requestRephrase: jest.fn(() =>
      Promise.resolve({
        results: WRITING_STYLES.reduce(
          (acc, style) => ({
            ...acc,
            [style]: `[${style}] response`,
          }),
          {} as Record<string, string>,
        ),
      }),
    ),
    requestRephraseStream: jest.fn(),
  };
});

const { requestRephraseStream } = jest.requireMock('../lib/api') as {
  requestRephraseStream: jest.Mock<
    Promise<void>,
    [unknown, StreamHandlers, AbortSignal | undefined]
  >;
};

function setupStreamMock() {
  const control: {
    handlers: StreamHandlers | null;
    resolve: () => void;
  } = { handlers: null, resolve: () => {} };

  requestRephraseStream.mockImplementation(
    async (
      _payload: unknown,
      handlers: StreamHandlers,
      signal?: AbortSignal,
    ) => {
      control.handlers = handlers;
      await new Promise<void>((resolve) => {
        control.resolve = resolve;
        signal?.addEventListener(
          'abort',
          () => resolve(),
          { once: true },
        );
      });
    },
  );

  return control;
}

function typeSampleText() {
  const textarea = screen.getByLabelText(/text to rephrase/i);
  return userEvent.type(textarea, 'Sample content to rephrase');
}

describe('App streaming behaviour', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('Process button disabled while streaming', async () => {
    const control = setupStreamMock();
    render(<App />);
    await typeSampleText();

    const processButton = screen.getByRole('button', { name: /process/i });
    const streamButton = screen.getByRole('button', { name: /stream/i });

    expect(processButton).toBeEnabled();
    await userEvent.click(streamButton);

    await waitFor(() => expect(processButton).toBeDisabled());

    // clean up by cancelling to resolve pending promise
    control.resolve();
  });

  test('streaming renders deltas as they arrive', async () => {
    const control = setupStreamMock();
    render(<App />);
    await typeSampleText();

    await userEvent.click(screen.getByRole('button', { name: /stream/i }));

    await act(async () => {
      control.handlers?.onChunk?.({ style: 'professional', delta: 'Hello', done: false });
      control.handlers?.onChunk?.({ style: 'professional', delta: 'world', done: true });
    });

    await waitFor(() =>
      expect(screen.getByText(/hello world/i, { selector: '.result-content p' })).toBeInTheDocument(),
    );

    await act(async () => {
      control.handlers?.onDone?.();
      control.resolve();
    });
  });

  test('cancel stops further streaming updates', async () => {
    const control = setupStreamMock();
    render(<App />);
    await typeSampleText();
    await userEvent.click(screen.getByRole('button', { name: /stream/i }));

    await act(async () => {
      control.handlers?.onChunk?.({ style: 'professional', delta: 'partial', done: false });
    });
    await waitFor(() =>
      expect(screen.getByText(/partial/i, { selector: '.result-content p' })).toBeInTheDocument(),
    );

    await userEvent.click(screen.getByRole('button', { name: /cancel/i }));
    await act(async () => {
      control.resolve();
      control.handlers?.onDone?.();
    });

    await act(async () => {
      control.handlers?.onChunk?.({ style: 'professional', delta: ' ignored', done: false });
    });

    expect(screen.getByText(/partial/i, { selector: '.result-content p' })).toHaveTextContent('partial');
  });
});
