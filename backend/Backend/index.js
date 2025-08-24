import 'dotenv/config';
import connectDB from "./config/db.js";
import express from "express";
import userRoutes from "./routers/userRoutes.js";
import creditProfileRoutes from './modals/creditProfileRoutes.js';
//import guidelineRoutes from "./routes/guidelineRoutes.js";

connectDB();

const app = express();

//app.use("/api/guidelines", guidelineRoutes);

app.use(express.json());

app.get("/", (req, res) => {
  res.send("API is running...");
});
app.use("/api/users", userRoutes);
app.use("/api/credit-profiles", creditProfileRoutes);
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
