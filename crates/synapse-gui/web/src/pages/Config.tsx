import { useEffect, useState } from 'react'
import { api } from '../api'

type FileKey = 'relays' | 'bootstrap' | 'models'
const FILES: { key: FileKey; label: string; help: string }[] = [
  { key: 'relays', label: 'relays.toml', help: 'Public relay peers. Multiaddrs ending in /p2p/<peer-id>.' },
  { key: 'bootstrap', label: 'bootstrap.toml', help: 'Kademlia DHT bootstrap peers.' },
  { key: 'models', label: 'models.toml', help: 'Default model id and shortcuts.' },
]

export function Config() {
  const [active, setActive] = useState<FileKey>('relays')
  const [contents, setContents] = useState('')
  const [original, setOriginal] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .getConfig(active)
      .then((r) => {
        if (!cancelled) {
          setContents(r.contents)
          setOriginal(r.contents)
          setErr(null)
        }
      })
      .catch((e) => !cancelled && setErr((e as Error).message))
    return () => {
      cancelled = true
    }
  }, [active])

  const dirty = contents !== original
  const save = async () => {
    setErr(null)
    try {
      await api.putConfig(active, contents)
      setOriginal(contents)
      setSavedAt(Date.now())
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  return (
    <div className="page">
      <h1>Config</h1>
      <div className="tabs">
        {FILES.map((f) => (
          <button
            key={f.key}
            className={active === f.key ? 'active' : ''}
            onClick={() => setActive(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>
      <p className="muted">{FILES.find((f) => f.key === active)?.help}</p>
      {err && <div className="error">{err}</div>}
      <textarea
        spellCheck={false}
        value={contents}
        onChange={(e) => setContents(e.target.value)}
        rows={20}
      />
      <div className="actions">
        <button onClick={save} disabled={!dirty}>
          save
        </button>
        <button onClick={() => setContents(original)} disabled={!dirty}>
          revert
        </button>
        {savedAt && <span className="muted">saved {new Date(savedAt).toLocaleTimeString()}</span>}
      </div>
    </div>
  )
}
