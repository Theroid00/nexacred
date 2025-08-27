import React, { useState } from "react";

const AuthModal = ({ isOpen, onClose, onLogin, onRegister }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    aadhaarNumber: ""
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLogin) {
      onLogin({ email: form.email, password: form.password });
    } else {
      onRegister(form);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
      background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
    }}>
      <div style={{
        background: '#18181b', color: '#fff', borderRadius: 16, padding: 32, minWidth: 340, boxShadow: '0 8px 32px rgba(0,0,0,0.25)', position: 'relative', maxWidth: 360, border: 'none'
      }}>
        <button onClick={onClose} style={{
          position: 'absolute', top: 12, right: 16, background: 'none', border: 'none', color: '#fff', fontSize: 28, cursor: 'pointer', fontWeight: 700
        }}>&times;</button>
        <div style={{ display: 'flex', marginBottom: 24, background: '#23232b', borderRadius: 8, overflow: 'hidden', boxShadow: 'none', border: 'none' }}>
          <button onClick={() => setIsLogin(true)} style={{
            flex: 1, padding: 12, border: 'none', borderRadius: '8px 0 0 8px', background: isLogin ? '#2563eb' : 'transparent', color: '#fff', fontWeight: 600, cursor: 'pointer', fontSize: 16, transition: 'background 0.2s'
          }}>Login</button>
          <button onClick={() => setIsLogin(false)} style={{
            flex: 1, padding: 12, border: 'none', borderRadius: '0 8px 8px 0', background: !isLogin ? '#2563eb' : 'transparent', color: '#fff', fontWeight: 600, cursor: 'pointer', fontSize: 16, transition: 'background 0.2s'
          }}>Register</button>
        </div>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {!isLogin && (
            <>
              <input name="username" placeholder="Username" value={form.username} onChange={handleChange} required style={{
                padding: 10, borderRadius: 8, border: '1px solid #333', background: '#23232b', color: '#fff', fontSize: 15
              }} />
              <input name="aadhaarNumber" placeholder="Aadhaar Number" value={form.aadhaarNumber} onChange={handleChange} required style={{
                padding: 10, borderRadius: 8, border: '1px solid #333', background: '#23232b', color: '#fff', fontSize: 15
              }} />
            </>
          )}
          <input name="email" placeholder="Email" value={form.email} onChange={handleChange} required style={{
            padding: 10, borderRadius: 8, border: '1px solid #333', background: '#23232b', color: '#fff', fontSize: 15
          }} />
          <input name="password" type="password" placeholder="Password" value={form.password} onChange={handleChange} required style={{
            padding: 10, borderRadius: 8, border: '1px solid #333', background: '#23232b', color: '#fff', fontSize: 15
          }} />
          <button type="submit" style={{
            marginTop: 8, padding: 12, borderRadius: 8, border: 'none', background: 'linear-gradient(90deg,#2563eb,#1e40af)', color: '#fff', fontWeight: 700, fontSize: 16, cursor: 'pointer', letterSpacing: 1
          }}>{isLogin ? "Login" : "Register"}</button>
        </form>
      </div>
    </div>
  );
};

export default AuthModal;
