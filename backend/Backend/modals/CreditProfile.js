import mongoose from "mongoose";

const creditProfileSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true // Faster queries for user profiles
  },

  // Basic details
  age: { type: Number, min: 18, max: 100, required: true },
  occupation: { type: String, required: true },

  // Financial details
  annualIncome: { type: Number, required: true, min: 0 },
  monthlyIncome: { 
    type: Number,
    required: true
  },
  numCreditCards: { type: Number, default: 0, min: 0 },

  // Loan-related
  loanTypes: [{ 
    type: String, 
    enum: ["personal", "home", "car", "education", "business", "other"] 
  }],
  activeLoans: { type: Number, default: 0 }, // track active loans count
  totalLoanAmount: { type: Number, default: 0 }, // sum of all loans

  // AI Features
  paymentHistory: { type: Number, default: 0, min: 0, max: 100 }, // % on-time payments
  creditUtilization: { type: Number, default: 0, min: 0, max: 100 }, // % utilization
  debtToIncomeRatio: { type: Number, default: 0, min: 0, max: 100 }, // Derived field
  savings: { type: Number, default: 0 }, // Optional, for ML

  // Audit
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: { type: Date, default: Date.now }
});

// 🔹 Pre-save hook to auto-calc monthlyIncome & debt ratio
creditProfileSchema.pre("save", function (next) {
  if (this.annualIncome && !this.monthlyIncome) {
    this.monthlyIncome = this.annualIncome / 12;
  }

  if (this.totalLoanAmount && this.annualIncome) {
    this.debtToIncomeRatio = Math.min(
      (this.totalLoanAmount / this.annualIncome) * 100,
      100
    );
  }

  this.updatedAt = Date.now();
  next();
});

export default mongoose.model("CreditProfile", creditProfileSchema);
