import Header from "./components/Header";
import React, { useState } from "react";
import AuthModal from "./components/AuthModal";
import Hero from "./components/Hero";
import TrustBadges from "./components/TrustBadges";
import Features from "./components/Features";
import HowItWorks from "./components/HowItWorks";
import WhyNexaCred from "./components/WhyNexaCred";
import CTA from "./components/CTA";
import Footer from "./components/Footer";

export default function App() {
  const [authOpen, setAuthOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token') || '');

  // Register API
  const handleRegister = async (form) => {
    try {
      const res = await fetch('/api/users/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      const data = await res.json();
      if (res.ok) {
        alert('Registration successful! Please login.');
        setAuthOpen(true);
      } else {
        alert(data.error || 'Registration failed');
      }
    } catch (err) {
      alert('Registration error');
    }
  };

  // Login API
  const handleLogin = async ({ email, password }) => {
    try {
      const res = await fetch('/api/users/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok) {
        setToken(data.token);
        localStorage.setItem('token', data.token);
        setUser(data.user);
        setAuthOpen(false);
      } else {
        alert(data.error || 'Login failed');
      }
    } catch (err) {
      alert('Login error');
    }
  };

  // Logout
  const handleLogout = () => {
    setUser(null);
    setToken('');
    localStorage.removeItem('token');
  };

  return (
    <>
      <AuthModal
        isOpen={authOpen}
        onClose={() => setAuthOpen(false)}
        onLogin={handleLogin}
        onRegister={handleRegister}
      />
      <main className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-gray-950 text-white">
        <Header user={user} handleLogout={handleLogout} setAuthOpen={setAuthOpen} />
        <Hero />
        <TrustBadges />
        <Features />
        <HowItWorks />
        <WhyNexaCred />
        <CTA />
        <Footer />
      </main>
    </>
  );
}
