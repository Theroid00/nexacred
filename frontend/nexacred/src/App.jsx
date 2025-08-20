import { Shield, TrendingUp, Users, Banknote, CheckCircle2, ArrowRight } from "lucide-react";

// NexaCred Landing Page (React + TailwindCSS)
// Pure JSX version (no TypeScript annotations)

export default function NexaCred() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-gray-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-black/40 bg-black/30 border-b border-white/10">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          <div className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">NexaCred</div>
          <nav className="hidden md:flex gap-6 text-gray-300">
            <a href="#features" className="hover:text-white">Features</a>
            <a href="#how" className="hover:text-white">How it works</a>
            <a href="#why" className="hover:text-white">Why NexaCred</a>
            <a href="#cta" className="hover:text-white">Get started</a>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16 text-center">
        <h1 className="text-4xl md:text-6xl font-extrabold leading-tight">
          Next-Gen Credit Scoring with <span className="bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">AI + Blockchain</span>
        </h1>
        <p className="mt-6 text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
          Fair. Fast. Fraud-proof. NexaCred analyzes real-time behavior and secures
          credit updates on an immutable ledger — unlocking inclusive finance and P2P lending.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <a href="#cta" className="inline-flex items-center gap-2 rounded-2xl bg-indigo-500 hover:bg-indigo-600 px-6 py-3 font-semibold">
            Request Demo <ArrowRight className="w-5 h-5" />
          </a>
          <a href="#how" className="inline-flex items-center gap-2 rounded-2xl border border-white/20 hover:border-white/40 px-6 py-3 font-semibold text-gray-200">
            See How it Works
          </a>
        </div>
      </section>

      {/* Trust badges */}
      <section className="mx-auto max-w-6xl px-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-400">
        <div className="rounded-xl border border-white/10 p-4 text-center">Real-time scoring</div>
        <div className="rounded-xl border border-white/10 p-4 text-center">Immutable records</div>
        <div className="rounded-xl border border-white/10 p-4 text-center">Inclusive data sources</div>
        <div className="rounded-xl border border-white/10 p-4 text-center">P2P-ready</div>
      </section>

      {/* Features */}
      <section id="features" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-3xl md:text-4xl font-bold text-center">Core Features</h2>
        <p className="text-gray-400 text-center mt-3 max-w-2xl mx-auto">
          Designed for transparency, speed, and security across modern financial ecosystems.
        </p>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          <Card icon={<TrendingUp className="w-10 h-10 text-indigo-400" />} title="AI-Powered Scoring"
                text="Real-time analysis of bank transactions, UPI & wallet activity, subscriptions, and e-commerce behaviors." />
          <Card icon={<Shield className="w-10 h-10 text-cyan-400" />} title="Blockchain Security"
                text="Scores are recorded as tamper-proof entries on a distributed ledger, verifiable by banks and fintechs." />
          <Card icon={<Users className="w-10 h-10 text-purple-400" />} title="P2P Lending Enablement"
                text="Verified scores empower direct lending decisions, lowering friction and expanding access." />
        </div>
      </section>

      {/* Diagram: How it works */}
      <section id="how" className="bg-black/30 border-y border-white/10">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-3xl md:text-4xl font-bold text-center">How NexaCred Works</h2>
          <p className="text-gray-400 text-center mt-3 max-w-2xl mx-auto">
            From multi-source data ingestion to on-chain verification — a streamlined, secure pipeline.
          </p>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-5 gap-4 items-stretch">
            <Step title="Data Sources" items={["Bank TXNs", "UPI & Wallets", "Subscriptions", "E-commerce"]} />
            <Arrow />
            <Step title="AI Scoring Engine" items={["ML Analysis", "Bias-aware Models", "Real-time Updates"]} />
            <Arrow />
            <Step title="Blockchain Layer" items={["Immutable Records", "Auditable", "Shared to Partners"]} />
          </div>

          <div className="mt-6 text-center text-gray-400 text-sm">
            * Partners include banks, fintech apps, and P2P lenders consuming verified scores.
          </div>
        </div>
      </section>

      {/* Why NexaCred */}
      <section id="why" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-3xl md:text-4xl font-bold text-center">Why NexaCred</h2>
        <div className="mt-10 grid md:grid-cols-3 gap-6">
          <Bullet text="Fairness by design — alternative data increases inclusion for thin-file users." />
          <Bullet text="Minutes, not months — scores reflect behavior changes in real time." />
          <Bullet text="Fraud resistance — verified identities and immutable ledgers reduce manipulation." />
        </div>
      </section>

      {/* CTA */}
      <section id="cta" className="mx-auto max-w-6xl px-6 pb-20">
        <div className="rounded-2xl border border-white/10 bg-gradient-to-r from-indigo-600/20 to-cyan-600/20 p-8 md:p-12 text-center">
          <h3 className="text-2xl md:text-3xl font-bold">Build on trust with NexaCred</h3>
          <p className="text-gray-300 mt-3 max-w-2xl mx-auto">
            Get a live demo and see how AI-driven, on-chain credit can power fair lending and risk decisions.
          </p>
          <div className="mt-6 flex justify-center gap-4">
            <a className="inline-flex items-center gap-2 rounded-2xl bg-indigo-500 hover:bg-indigo-600 px-6 py-3 font-semibold" href="#">
              Request Demo <ArrowRight className="w-5 h-5" />
            </a>
            <a className="inline-flex items-center gap-2 rounded-2xl border border-white/20 hover:border-white/40 px-6 py-3 font-semibold" href="#features">
              Explore Features
            </a>
          </div>
          <ul className="mt-6 flex flex-wrap justify-center gap-4 text-sm text-gray-400">
            <li className="inline-flex items-center gap-2"><CheckCircle2 className="w-4 h-4"/> Real-time</li>
            <li className="inline-flex items-center gap-2"><CheckCircle2 className="w-4 h-4"/> Auditable</li>
            <li className="inline-flex items-center gap-2"><CheckCircle2 className="w-4 h-4"/> Developer-friendly</li>
          </ul>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center text-gray-400 border-t border-white/10">
        © {new Date().getFullYear()} NexaCred — Fair. Fast. Fraud-proof.
      </footer>
    </main>
  );
}

// Components rewritten without TS annotations
function Card({ icon, title, text }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 hover:bg-white/[0.06] transition">
      <div className="mb-3">{icon}</div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="text-gray-400 mt-2 text-sm leading-relaxed">{text}</p>
    </div>
  );
}

function Step({ title, items }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
      <div className="text-sm uppercase tracking-wider text-gray-400">{title}</div>
      <ul className="mt-3 space-y-2 text-sm">
        {items.map((it, i) => (
          <li key={i} className="inline-flex items-center gap-2">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-gradient-to-r from-indigo-400 to-cyan-400" />
            <span className="text-gray-300">{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Arrow() {
  return (
    <div className="hidden md:flex items-center justify-center">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
        <path d="M4 12h13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M13 7l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    </div>
  );
}

function Bullet({ text }) {
  return (
    <div className="rounded-2xl border border-white/10 p-6 bg-white/[0.03]">
      <div className="inline-flex items-center gap-2 text-gray-200">
        <Banknote className="w-5 h-5 text-emerald-400" />
        <span className="font-medium">{text}</span>
      </div>
    </div>
  );
}
