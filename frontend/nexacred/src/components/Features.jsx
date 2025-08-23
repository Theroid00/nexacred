import { TrendingUp, Shield, Users } from "lucide-react";
import Card from "./Card";

export default function Features() {
  return (
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
  );
}
