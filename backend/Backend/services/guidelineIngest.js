// import pdfParse from "pdf-parse";
// import fs from "fs";
// import Guideline from "./models/guideline.js";
// import { pipeline } from "@xenova/transformers"; // HuggingFace embeddings in Node.js

// async function processPDF() {
//   const docId = "sbi-guidelines-v1"; 

//   // 1️⃣ Check if already stored
//   const existing = await Guideline.findOne({ docId });
//   if (existing) {
//     console.log("⚠️ SBI Guidelines already stored in MongoDB. Skipping...");
//     return;
//   }

//   // 2️⃣ Load PDF
//   const dataBuffer = fs.readFileSync("./sbi-guidelines.pdf");
//   const pdfData = await pdfParse(dataBuffer);
//   const text = pdfData.text;

//   // 3️⃣ Split text into chunks
//   const chunks = text.match(/(.|[\r\n]){1,500}/g);

//   // 4️⃣ Load embedding model (e.g. `all-MiniLM-L6-v2`)
//   const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

//   // 5️⃣ Process each chunk
//   for (let i = 0; i < chunks.length; i++) {
//     const chunk = chunks[i];
    
//     // Generate embedding (vector)
//     const embedding = await embedder(chunk, { pooling: "mean", normalize: true });
    
//     // Save in DB
//     await Guideline.create({
//       docId,
//       chunk,
//       chunkIndex: i,
//       embedding: Array.from(embedding.data[0]) // Convert tensor to plain array
//     });
//   }

//   console.log("✅ SBI Guidelines stored in MongoDB with custom embeddings!");
// }

// processPDF();
