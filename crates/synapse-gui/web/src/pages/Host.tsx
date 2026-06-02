import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import { api, StatusResponse } from '../api'
import { useChatStream, UiChatEvent, WsStatus } from '../ws'

export function Host() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [model, setModel] = useState<string>('')
  const [maxTokens, setMaxTokens] = useState(200)
  const [listen, setListen] = useState('0.0.0.0:0')
  const [pairingCode, setPairingCode] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [sendNote, setSendNote] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Poll status to discover cached models and reflect stop/start state.
  useEffect(() => {
    let cancelled = false
    const tick = async (): Promise<void> => {
      try {
        const s = await api.status()
        if (!cancelled) {
          setStatus(s)
          if (!model && s.cached_models[0]) setModel(s.cached_models[0])
        }
      } catch (e) {
        if (!cancelled) setErr(errorMessage(e))
      }
    }
    void tick()
    const t = setInterval(() => void tick(), 2000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [model])

  const isMaster = status?.session.role === 'master'
  const { status: wsStatus, events, sendPrompt, reset } = useChatStream(isMaster)

  // When a master session appears, clear any stale events from before.
  useEffect(() => {
    if (isMaster) reset()
  }, [isMaster, reset])

  // Pull the pairing code from the polled status so the UI always
  // shows the canonical value (in case of reconnect, etc).
  useEffect(() => {
    if (isMaster && status?.session.pairing_code) {
      setPairingCode(status.session.pairing_code)
    } else if (!isMaster) {
      setPairingCode(null)
    }
  }, [isMaster, status?.session.pairing_code])

  // Auto-scroll the chat on new events.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [events])

  const assistantText = useMemo(() => renderEvents(events), [events])

  const start = useCallback(async (): Promise<void> => {
    setBusy(true)
    setErr(null)
    setSendNote(null)
    try {
      const { pairing_code, session } = await api.startHost(
        model,
        maxTokens,
        listen,
      )
      setPairingCode(pairing_code)
      // Immediately reflect the new session in our status cache so
      // `isMaster` flips true on the next render — without this the
      // WS waits up to 2s for the polling tick, and any prompt the
      // user types in that window would be silently dropped.
      setStatus((prev) =>
        prev ? { ...prev, session } : prev,
      )
    } catch (e) {
      setErr(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }, [model, maxTokens, listen])

  const stop = useCallback(async (): Promise<void> => {
    setBusy(true)
    try {
      await api.stop()
    } catch (e) {
      setErr(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const onSubmit = useCallback(
    (e: FormEvent): void => {
      e.preventDefault()
      const text = input.trim()
      if (!text) return
      const queued = sendPrompt(text)
      if (queued) {
        setInput('')
        setSendNote(null)
      } else {
        // WS is closed and not connecting — surface the problem so
        // the user understands why nothing happened.
        setSendNote(
          `couldn't send: websocket is ${wsStatus}. try stopping and re-starting the host.`,
        )
      }
    },
    [input, sendPrompt, wsStatus],
  )

  if (err && !status) return <div className="error">error: {err}</div>
  if (!status) return <div className="loading">loading…</div>

  if (!isMaster) {
    return <StartForm
      model={model}
      setModel={setModel}
      maxTokens={maxTokens}
      setMaxTokens={setMaxTokens}
      listen={listen}
      setListen={setListen}
      cachedModels={status.cached_models}
      busy={busy}
      err={err}
      onStart={() => void start()}
    />
  }

  return <ActiveMaster
    pairingCode={pairingCode}
    wsStatus={wsStatus}
    assistantText={assistantText}
    scrollRef={scrollRef}
    input={input}
    setInput={setInput}
    sendNote={sendNote}
    busy={busy}
    onSubmit={onSubmit}
    onStop={() => void stop()}
  />
}

type StartFormProps = {
  model: string
  setModel: (v: string) => void
  maxTokens: number
  setMaxTokens: (n: number) => void
  listen: string
  setListen: (v: string) => void
  cachedModels: string[]
  busy: boolean
  err: string | null
  onStart: () => void
}

function StartForm(props: StartFormProps) {
  return (
    <div className="page">
      <h1>Host (master)</h1>
      {props.err && <div className="error">{props.err}</div>}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          props.onStart()
        }}
        className="form"
      >
        <label>
          <span>model</span>
          <input
            list="models"
            value={props.model}
            onChange={(e) => props.setModel(e.target.value)}
            placeholder="hf-internal-testing/tiny-random-LlamaForCausalLM"
            required
          />
          <datalist id="models">
            {props.cachedModels.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </label>
        <label>
          <span>max tokens</span>
          <input
            type="number"
            min={1}
            max={4096}
            value={props.maxTokens}
            onChange={(e) => props.setMaxTokens(parseInt(e.target.value, 10) || 0)}
          />
        </label>
        <label>
          <span>listen address</span>
          <input value={props.listen} onChange={(e) => props.setListen(e.target.value)} />
        </label>
        <button type="submit" disabled={props.busy || !props.model}>
          {props.busy ? 'starting…' : 'start hosting'}
        </button>
      </form>
    </div>
  )
}

type ActiveMasterProps = {
  pairingCode: string | null
  wsStatus: WsStatus
  assistantText: string
  scrollRef: React.RefObject<HTMLDivElement>
  input: string
  setInput: (v: string) => void
  sendNote: string | null
  busy: boolean
  onSubmit: (e: FormEvent) => void
  onStop: () => void
}

function ActiveMaster(props: ActiveMasterProps) {
  const wsBadge = `ws: ${props.wsStatus}`
  const canSend = props.wsStatus === 'open' || props.wsStatus === 'connecting'
  return (
    <div className="page">
      <h1>Host (master)</h1>
      <div className="pairing">
        <span className="label">pairing code</span>
        <code className="big">{props.pairingCode ?? '—'}</code>
        <span className={`muted ws-${props.wsStatus}`}>{wsBadge}</span>
        <button className="danger" onClick={props.onStop} disabled={props.busy}>
          stop
        </button>
      </div>

      <div className="chat" ref={props.scrollRef}>
        <pre className="assistant">{props.assistantText || ' '}</pre>
      </div>

      {props.sendNote && <div className="error">{props.sendNote}</div>}

      <form onSubmit={props.onSubmit} className="prompt">
        <input
          autoFocus
          value={props.input}
          onChange={(e) => props.setInput(e.target.value)}
          placeholder="type a prompt and press Enter…"
        />
        <button type="submit" disabled={!canSend || !props.input.trim()}>
          send
        </button>
      </form>
    </div>
  )
}

function renderEvents(events: UiChatEvent[]): string {
  let out = ''
  for (const e of events) {
    if (e.type === 'token') {
      out += e.text
    } else if (e.type === 'generation_done') {
      out += `\n\n[done — ${e.tokens_generated} tokens, ${e.reason}]`
    }
  }
  return out
}

function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message
  return 'unexpected error'
}
