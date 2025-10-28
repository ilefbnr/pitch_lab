import express from "express";
import bcrypt from "bcryptjs";
import db from "../db.js";

const router = express.Router();

// ===== SIGNUP =====
router.post("/signup", async (req, res) => {
  console.log("Incoming signup request body:", req.body); // <- log input

  const { fullName, email, password } = req.body;

  if (!fullName || !email || !password) {
    return res.status(400).json({ detail: "All fields are required." });
  }

  try {
    // Check if email exists
    db.query("SELECT * FROM users WHERE email = ?", [email], async (err, results) => {
      if (err) {
        console.error("DB error:", err); // <- log DB errors
        return res.status(500).json({ detail: "Database error." });
      }
      if (results.length > 0) {
        return res.status(400).json({ detail: "Email already registered." });
      }

      const hashedPassword = await bcrypt.hash(password, 10);

      db.query(
        "INSERT INTO users (full_name, email, password) VALUES (?, ?, ?)",
        [fullName, email, hashedPassword],
        (err) => {
          if (err) {
            console.error("Insert DB error:", err); // <- log DB errors
            return res.status(500).json({ detail: "Signup failed." });
          }
          res.status(201).json({ message: "User registered successfully!" });
        }
      );
    });
  } catch (error) {
    console.error("Unhandled error:", error); // <- catch all
    res.status(500).json({ detail: "Server error." });
  }
});


// ===== LOGIN =====
router.post("/login", (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ detail: "Email and password required." });
  }

  db.query("SELECT * FROM users WHERE email = ?", [email], async (err, results) => {
    if (err) return res.status(500).json({ detail: "Database error." });
    if (results.length === 0) {
      return res.status(401).json({ detail: "Invalid credentials." });
    }

    const user = results[0];
    const passwordMatch = await bcrypt.compare(password, user.password);

    if (!passwordMatch) {
      return res.status(401).json({ detail: "Invalid credentials." });
    }

    res.json({
      message: "Login successful!",
      user: {
        id: user.id,
        fullName: user.full_name,
        email: user.email,
      },
    });
  });
});

export default router;
