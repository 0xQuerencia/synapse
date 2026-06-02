// WebSocket client for live chat events from the master session.
//
// Wire format: see crates/synapse-gui/src/ws.rs.
//
// The hook buffers prompts typed while the WS is still connecting,
// and flushes them on `open`. This avoids a real race: when the
// React side calls `startHost`, the session exists server-side
// immediately, but `isMaster` only flips after a 2s status poll —
// typing during that window would otherwise be dropped silently.

import { useEffect, useRef, useState } from 'react'

export type UiChatEvent =
  | { type: 'worker_connected' }
  | { type: 'token'; text: string }
  | { type: 'generation_done'; tokens_generated: number; reason: string }

interface ServerEvent {
  type: 'event'
  id: string | null
  event: UiChatEvent
}

interface ServerHello {
  type: 'hello'
  role: string
}

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface UseChatStream {
  status: WsStatus
  events: UiChatEvent[]
  /** Send a prompt. Returns true if it was queued, false if dropped. */
  sendPrompt: (text: string) => boolean
  /** Clear accumulated events (e.g. when starting a new session). */
  reset: () => void
}

export function useChatStream(enabled: boolean): UseChatStream {
  const [status, setStatus] = useState<WsStatus>('closed')
  const [events, setEvents] = useState<UiChatEvent[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  // Prompts typed before the WS finishes connecting. Flushed on open.
  const pendingRef = useRef<string[]>([])

  useEffect(() => {
    if (!enabled) {
      setStatus('closed')
      setEvents([])
      pendingRef.current = []
      wsRef.current?.close()
      wsRef.current = null
      return
    }
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${window.location.host}/ws`
    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('open')
      // Flush any prompts the user typed while we were connecting.
      const queue = pendingRef.current
      if (queue.length) {
        pendingRef.current = []
        for (const text of queue) {
          ws.send(
            JSON.stringify({
              type: 'prompt',
              id: makeId(),
              text,
            }),
          )
        }
      }
    }
    ws.onerror = () => setStatus('error')
    ws.onclose = () => setStatus('closed')
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as ServerEvent | ServerHello
        if (data.type === 'event') {
          setEvents((prev) => [...prev, data.event])
        }
      } catch (err) {
        // Surface parsing issues in the dev console — they indicate a
        // wire-format mismatch between the Rust ws.rs and this client.
        console.warn('[synapse-gui ws] failed to parse frame', err, msg.data)
      }
    }

    return () => {
      ws.close()
    }
  }, [enabled])

  const sendPrompt = (text: string): boolean => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'prompt',
          id: makeId(),
          text,
        }),
      )
      return true
    }
    // Buffer until the WS is open, so prompts typed immediately
    // after `Start hosting` aren't lost to the 2s status poll race.
    if (status === 'connecting') {
      pendingRef.current.push(text)
      return true
    }
    return false
  }

  const reset = () => {
    setEvents([])
    pendingRef.current = []
  }

  return { status, events, sendPrompt, reset }
}

// UUID-like ID without `crypto.randomUUID()`, which is only available
// in secure contexts. The id is just a correlation hint for future
// per-prompt event multiplexing, so a timestamped nonce is fine.
function makeId(): string {
  const t = Date.now().toString(36)
  const r = Math.random().toString(36).slice(2, 10)
  return `${t}-${r}`
}
