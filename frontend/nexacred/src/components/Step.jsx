export default function Step({ title, items }) {
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
