import { ArrowRight } from "lucide-react";

export default function Hero() {
  return (
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
  );
}
