import Bullet from "./Bullet";

export default function WhyNexaCred() {
  return (
    <section id="why" className="mx-auto max-w-6xl px-6 py-16">
      <h2 className="text-3xl md:text-4xl font-bold text-center">Why NexaCred</h2>
      <div className="mt-10 grid md:grid-cols-3 gap-6">
        <Bullet text="Fairness by design — alternative data increases inclusion for thin-file users." />
        <Bullet text="Minutes, not months — scores reflect behavior changes in real time." />
        <Bullet text="Fraud resistance — verified identities and immutable ledgers reduce manipulation." />
      </div>
    </section>
  );
}
