import { Banknote } from "lucide-react";

export default function Bullet({ text }) {
  return (
    <div className="rounded-2xl border border-white/10 p-6 bg-white/[0.03]">
      <div className="inline-flex items-center gap-2 text-gray-200">
        <Banknote className="w-5 h-5 text-emerald-400" />
        <span className="font-medium">{text}</span>
      </div>
    </div>
  );
}
