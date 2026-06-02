import { useEffect, useState } from 'react'
import { api, StatusResponse } from '../api'

export function Relay() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [listen, setListen] = useState('0.0.0.0:4001')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const s = await api.status()
        if (!cancelled) setStatus(s)
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
  }, [])

  const start = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    try {
      const { session } = await api.startRelay(listen)
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

  const isRelay = status.session.role === 'relay'

  return (
    <div className="page">
      <h1>Relay server</h1>

      {!isRelay ? (
        <form onSubmit={start} className="form">
          <label>
            <span>listen address</span>
            <input
              value={listen}
              onChange={(e) => setListen(e.target.value)}
              placeholder="0.0.0.0:4001"
            />
          </label>
          <p className="muted">
            For a public relay, run on a VPS with a public IP and add the printed
            peer id to <code>config/relays.toml</code> on master/worker machines.
          </p>
          <button type="submit" disabled={busy}>
            {busy ? 'starting…' : 'start relay'}
          </button>
        </form>
      ) : (
        <div className="status">
          <p>running as relay.</p>
          {status.session.peer_id && (
            <p>
              peer id: <code>{status.session.peer_id}</code>
            </p>
          )}
          {status.session.listen_addr && (
            <p>
              listening on: <code>{status.session.listen_addr}</code>
            </p>
          )}
          <button className="danger" onClick={stop} disabled={busy}>
            stop
          </button>
        </div>
      )}
    </div>
  )
}
