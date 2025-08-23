export default function Card({ icon, title, text }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 hover:bg-white/[0.06] transition">
      <div className="mb-3">{icon}</div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="text-gray-400 mt-2 text-sm leading-relaxed">{text}</p>
    </div>
  );
}
