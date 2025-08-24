// // /controllers/guidelineController.js
// import Guideline from "../models/guideline.js";
// import { pipeline } from "@xenova/transformers";

// // Load embeddings once
// const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

// export async function queryGuidelines(req, res) {
//   try {
//     const { question } = req.body;

//     // 1️⃣ Embed the user query
//     const queryEmbedding = await embedder(question, { pooling: "mean", normalize: true });
//     const queryVector = Array.from(queryEmbedding.data[0]);

//     // 2️⃣ Fetch all stored chunks from DB
//     const allChunks = await Guideline.find();

//     // 3️⃣ Compute cosine similarity manually
//     function cosineSim(a, b) {
//       const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
//       const magA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
//       const magB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
//       return dot / (magA * magB);
//     }

//     const ranked = allChunks
//       .map(doc => ({
//         ...doc._doc,
//         score: cosineSim(queryVector, doc.embedding)
//       }))
//       .sort((a, b) => b.score - a.score);

//     // 4️⃣ Return top 3 matches
//     res.json({ results: ranked.slice(0, 3) });

//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ error: "Something went wrong" });
//   }
// }
