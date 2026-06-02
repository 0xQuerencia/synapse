import { useEffect, useState } from 'react'
import { api, StatusResponse } from '../api'

export function Join() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [model, setModel] = useState('')
  const [code, setCode] = useState('')
  const [masterAddr, setMasterAddr] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const s = await api.status()
        if (!cancelled) {
          setStatus(s)
          if (!model && s.cached_models[0]) setModel(s.cached_models[0])
        }
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    }
    tick()
    const t = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [model])

  const join = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    try {
      const { session } = await api.startJoin(
        model,
        code,
        masterAddr.trim() || undefined,
      )
      setStatus((prev) => (prev ? { ...prev, session } : prev))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      await api.stop()
    } finally {
      setBusy(false)
    }
  }

  if (err) return <div className="error">error: {err}</div>
  if (!status) return <div className="loading">loading…</div>

  const isWorker = status.session.role === 'worker'

  return (
    <div className="page">
      <h1>Join (worker)</h1>

      {!isWorker ? (
        <form onSubmit={join} className="form">
          <label>
            <span>model</span>
            <input
              list="models"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="hf-internal-testing/tiny-random-LlamaForCausalLM"
              required
            />
            <datalist id="models">
              {status.cached_models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
          <label>
            <span>pairing code</span>
            <input
              value={code}
              onChange={(e) => setCode(e.target.value)}
              required
            />
          </label>
          <label>
            <span>master address (optional, mDNS if blank)</span>
            <input
              value={masterAddr}
              onChange={(e) => setMasterAddr(e.target.value)}
              placeholder="/ip4/192.168.1.10/udp/4001/quic-v1"
            />
          </label>
          <button type="submit" disabled={busy || !model || !code}>
            {busy ? 'connecting…' : 'connect'}
          </button>
        </form>
      ) : (
        <div className="status">
          <p>running as worker.</p>
          {status.session.peer_id && (
            <p>
              peer id: <code>{status.session.peer_id}</code>
            </p>
          )}
          <p className="muted">
            Watch the host's chat to confirm the worker contributes. Use stop to disconnect.
          </p>
          <button className="danger" onClick={stop} disabled={busy}>
            stop
          </button>
        </div>
      )}
    </div>
  )
}
