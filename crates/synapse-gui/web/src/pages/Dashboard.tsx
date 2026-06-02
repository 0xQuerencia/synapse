import { useEffect, useState } from 'react'
import { api, Identity, StatusResponse } from '../api'

export function Dashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [identity, setIdentity] = useState<Identity | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const [s, i] = await Promise.all([api.status(), api.identity()])
        if (!cancelled) {
          setStatus(s)
          setIdentity(i)
          setErr(null)
        }
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    }
    refresh()
    const t = setInterval(refresh, 2000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [])

  if (err) return <div className="error">error: {err}</div>
  if (!status || !identity) return <div className="loading">loading…</div>

  const role = status.session.role

  return (
    <div className="page">
      <h1>Dashboard</h1>
      <div className="cards">
        <Card title="Identity">
          <KV k="peer id" v={status.peer_id} mono />
          <KV k="short id" v={identity.short_id} mono />
          <KV k="public key" v={identity.public_key} mono />
        </Card>

        <Card title="Session">
          <div className="kv">
            <span className="k">status</span>
            <span className={`v badge badge-${role}`}>{role}</span>
          </div>
          {status.session.peer_id && <KV k="peer id" v={status.session.peer_id} mono />}
          {status.session.model_id && <KV k="model" v={status.session.model_id} />}
          {status.session.pairing_code && (
            <KV k="pairing code" v={status.session.pairing_code} mono />
          )}
          {status.session.listen_addr && <KV k="listen" v={status.session.listen_addr} mono />}
          {role !== 'idle' && (
            <button className="danger" onClick={() => api.stop()}>
              stop session
            </button>
          )}
        </Card>

        <Card title="Network">
          <KV k="bootstrap peers" v={String(status.bootstrap_count)} />
          <KV k="relay servers" v={String(status.relay_count)} />
        </Card>

        <Card title="Cached models">
          {status.cached_models.length === 0 ? (
            <em>none yet — run a demo or host to download one.</em>
          ) : (
            <ul className="list">
              {status.cached_models.map((m) => (
                <li key={m}><code>{m}</code></li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function KV({ k, v, mono = false }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className={`v${mono ? ' mono' : ''}`}>{v}</span>
    </div>
  )
}
