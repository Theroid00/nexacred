import mongoose from "mongoose";

const creditScoreSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true // faster lookup by user
  },

  // Score Info
  score: { type: Number, required: true, min: 300, max: 850 },
  category: { 
    type: String, 
    enum: ["Poor", "Fair", "Good", "Very Good", "Excellent"], 
    required: true 
  },

  // AI / ML Info
  mlVersion: { type: String, default: "v1.0" }, // model version
  confidence: { type: Number, min: 0, max: 1, default: 0.85 }, // ML confidence
  featuresUsed: [{ type: String }], // track which features were used in scoring
  reasonCodes: [{ type: String }], // e.g., ["High utilization", "Low income"]

  // Blockchain / Transparency
  blockchainTxHash: { type: String },
  verifiedOnChain: { type: Boolean, default: false },

  // Risk / Flags
  riskLevel: { 
    type: String, 
    enum: ["Low", "Medium", "High"], 
    default: "Medium" 
  },
  flagged: { type: Boolean, default: false }, // suspicious activity flag

  // Audit Trail
  scoreHistory: [
    {
      score: Number,
      category: String,
      timestamp: { type: Date, default: Date.now }
    }
  ],

  // Timestamps
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// 🔹 Pre-save hook: Push score to history on change
creditScoreSchema.pre("save", function (next) {
  if (this.isModified("score") || this.isModified("category")) {
    this.scoreHistory.push({
      score: this.score,
      category: this.category
    });
  }
  this.updatedAt = Date.now();
  next();
});

export default mongoose.model("CreditScore", creditScoreSchema);
