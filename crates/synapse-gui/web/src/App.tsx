import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { Host } from './pages/Host'
import { Join } from './pages/Join'
import { Relay } from './pages/Relay'
import { Config } from './pages/Config'

export default function App() {
  return (
    <div className="app">
      <Nav />
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/host" element={<Host />} />
          <Route path="/join" element={<Join />} />
          <Route path="/relay" element={<Relay />} />
          <Route path="/config" element={<Config />} />
        </Routes>
      </main>
    </div>
  )
}

function Nav() {
  const loc = useLocation()
  const links = [
    { to: '/', label: 'Dashboard' },
    { to: '/host', label: 'Host' },
    { to: '/join', label: 'Join' },
    { to: '/relay', label: 'Relay' },
    { to: '/config', label: 'Config' },
  ]
  return (
    <header className="nav">
      <div className="brand">Synapse GUI</div>
      <nav>
        {links.map((l) => (
          <Link
            key={l.to}
            to={l.to}
            className={loc.pathname === l.to ? 'active' : ''}
          >
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  )
}
