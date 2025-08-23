import Header from "./components/Header";
import Hero from "./components/Hero";
import TrustBadges from "./components/TrustBadges";
import Features from "./components/Features";
import HowItWorks from "./components/HowItWorks";
import WhyNexaCred from "./components/WhyNexaCred";
import CTA from "./components/CTA";
import Footer from "./components/Footer";

export default function App() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-gray-950 text-white">
      <Header />
      <Hero />
      <TrustBadges />
      <Features />
      <HowItWorks />
      <WhyNexaCred />
      <CTA />
      <Footer />
    </main>
  );
}
