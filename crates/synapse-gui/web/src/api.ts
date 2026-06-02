// Typed REST client. All paths are relative so the same bundle
// works regardless of which port the Rust server is on.

export type SessionRole = 'idle' | 'master' | 'worker' | 'relay'

export interface Identity {
  peer_id: string
  short_id: string
  public_key: string
  key_path: string
}

export interface SessionSnapshot {
  role: SessionRole
  peer_id: string | null
  model_id: string | null
  pairing_code: string | null
  max_tokens: number | null
  listen_addr: string | null
}

export interface StatusResponse {
  peer_id: string
  cached_models: string[]
  bootstrap_count: number
  relay_count: number
  session: SessionSnapshot
}

export interface ConfigResponse {
  file: string
  contents: string
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) msg = body.error
    } catch {
      // ignore
    }
    throw new ApiError(res.status, msg)
  }
  return (await res.json()) as T
}

export const api = {
  identity: () => request<Identity>('/api/identity'),
  status: () => request<StatusResponse>('/api/status'),
  cachedModels: () => request<string[]>('/api/cached-models'),
  getConfig: (file: 'relays' | 'bootstrap' | 'models') =>
    request<ConfigResponse>(`/api/config/${file}`),
  putConfig: (file: 'relays' | 'bootstrap' | 'models', contents: string) =>
    request<ConfigResponse>(`/api/config/${file}`, {
      method: 'PUT',
      body: JSON.stringify({ contents }),
    }),

  startHost: (model: string, max_tokens: number, listen: string) =>
    request<{ pairing_code: string; session: SessionSnapshot }>(
      '/api/session/host',
      {
        method: 'POST',
        body: JSON.stringify({ model, max_tokens, listen }),
      },
    ),
  startJoin: (model: string, code: string, master_addr?: string) =>
    request<{ peer_id: string; session: SessionSnapshot }>('/api/session/join', {
      method: 'POST',
      body: JSON.stringify({ model, code, master_addr: master_addr ?? null }),
    }),
  startRelay: (listen: string) =>
    request<{ peer_id: string; session: SessionSnapshot }>('/api/session/relay', {
      method: 'POST',
      body: JSON.stringify({ listen }),
    }),
  stop: () => request<{ ok: true }>('/api/session/stop', { method: 'POST' }),
}
