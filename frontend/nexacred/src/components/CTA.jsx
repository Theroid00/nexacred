import { ArrowRight, CheckCircle2 } from "lucide-react";

export default function CTA() {
  return (
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
  );
}
