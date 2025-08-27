export default function Header({ user, handleLogout, setAuthOpen }) {
  return (
    <header className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-black/40 bg-black/30 border-b border-white/10">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
        <div className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">NexaCred</div>
        <nav className="hidden md:flex gap-6 text-gray-300 items-center">
          <a href="#features" className="hover:text-white">Features</a>
          <a href="#how" className="hover:text-white">How it works</a>
          <a href="#why" className="hover:text-white">Why NexaCred</a>
          <a href="#cta" className="hover:text-white">Get started</a>
          {user ? (
            <>
              <span style={{ marginLeft: 24, marginRight: 8, color: '#fff', fontWeight: 500, fontSize: 15 }}>Welcome, {user.username || user.email}</span>
              <button onClick={handleLogout} style={{ background: '#18181b', color: '#fff', border: '1px solid #333', borderRadius: 8, padding: '6px 18px', fontWeight: 600, cursor: 'pointer' }}>Logout</button>
            </>
          ) : (
            <button onClick={() => setAuthOpen(true)} style={{ background: '#18181b', color: '#fff', border: '1px solid #333', borderRadius: 8, padding: '6px 18px', fontWeight: 600, cursor: 'pointer', marginLeft: 24 }}>Register / Login</button>
          )}
        </nav>
      </div>
    </header>
  );
}
