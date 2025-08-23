import 'dotenv/config';
import connectDB from "./config/db.js";
import express from "express";

connectDB();

const app = express();

app.use(express.json());

app.get("/", (req, res) => {
  res.send("API is running...");
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
