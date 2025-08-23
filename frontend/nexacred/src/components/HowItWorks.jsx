import Step from "./Step";
import Arrow from "./Arrow";

export default function HowItWorks() {
  return (
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
  );
}
