// /controllers/userController.js
import User from '../modals/User.js';
import bcrypt from "bcryptjs";  
import jwt from "jsonwebtoken";
import dotenv from "dotenv";
dotenv.config();

// 1️⃣ Register a new user
export async function registerUser(req, res) {
  try {
    const { username, email, password, aadhaarNumber } = req.body;

    // Check if user exists
    const existing = await User.findOne({ $or: [{ email }, { username }] });
    if (existing) {
      return res.status(400).json({ error: "Username or Email already exists" });
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const passwordHash = await bcrypt.hash(password, salt);

    // Save user
    const user = await User.create({
      username,
      email,
      passwordHash,
      aadhaarNumber
    });

    res.status(201).json({ message: "User registered successfully", user });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Server error" });
  }
}

// 2️⃣ Login user
export async function loginUser(req, res) {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ error: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, user.passwordHash);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    const token = jwt.sign(
      { userId: user._id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: "1h" }
    );

    // Optionally, save the token in the database if you want to track logins
    // user.token = token;
    // await user.save();

     res.json({ token, user });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}

// 3️⃣ Get all users
export async function getUsers(req, res) {
  try {
    const users = await User.find().select("-passwordHash"); // hide password
    res.json(users);
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}

// 4️⃣ Get user by ID
export async function getUserById(req, res) {
  try {
    const user = await User.findById(req.params.id).select("-passwordHash");
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json(user);
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}

// 5️⃣ Update user
export async function updateUser(req, res) {
  try {
    const { username, email, aadhaarNumber } = req.body;
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { username, email, aadhaarNumber },
      { new: true }
    ).select("-passwordHash");

    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ message: "User updated", user });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}

// 6️⃣ Delete user
export async function deleteUser(req, res) {
  try {
    const user = await User.findByIdAndDelete(req.params.id);
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ message: "User deleted" });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}
