process.env.NODE_NO_WARNINGS = '1';
require("dotenv").config();
const express  = require("express");
const multer   = require("multer");
const path     = require("path");
const fs       = require("fs");
const axios    = require("axios");
const mongoose = require("mongoose");

const app  = express();
const PORT = 3000;

// ── MongoDB ────────────────────────────────────────────────────────────────
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log("☁️  Connected to MongoDB Atlas"))
  .catch(err => console.error("❌ DB Error:", err));

const sessionSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true },
  fileName:  String,
  content:   String,
  title:     String,
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
  history: [{
    role:      String,
    text:      String,
    timestamp: { type: Date, default: Date.now }
  }]
});
const Session = mongoose.model("Session", sessionSchema);

// ── Config ─────────────────────────────────────────────────────────────────
const OLLAMA_API = "http://localhost:11434/api/generate";
const MODEL_NAME = "llama3.2:1b";

// Character limits per feature — keeps prompts short and the model fast
const LIMITS = { ask: 4000, summary: 6000, quiz: 5000, flashcards: 4000 };

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = path.join(__dirname, "uploads");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => cb(null, Date.now() + "_" + file.originalname),
});
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json({ limit: "10mb" }));
app.use(express.static(__dirname));

// ── Streaming helper ───────────────────────────────────────────────────────
// Sends tokens to the browser as Server-Sent Events the moment Ollama produces them.
// The user sees words appear in real time instead of waiting for the full response.
async function streamOllama(prompt, res) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const ollamaRes = await axios.post(
    OLLAMA_API,
    { model: MODEL_NAME, prompt, stream: true },
    { responseType: "stream", timeout: 120000 }
  );

  let fullText = "";

  await new Promise((resolve, reject) => {
    ollamaRes.data.on("data", chunk => {
      const lines = chunk.toString().split("\n").filter(Boolean);
      for (const line of lines) {
        try {
          const json = JSON.parse(line);
          if (json.response) {
            fullText += json.response;
            res.write(`data: ${JSON.stringify({ token: json.response })}\n\n`);
          }
          if (json.done) {
            res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
          }
        } catch {}
      }
    });
    ollamaRes.data.on("end", resolve);
    ollamaRes.data.on("error", reject);
  });

  return fullText;
}

// ── Routes ─────────────────────────────────────────────────────────────────
app.get("/ollama-status", (req, res) => res.json({ running: true, hasModel: true }));

app.get("/sessions", async (req, res) => {
  try {
    const sessions = await Session.find(
      { "history.0": { $exists: true } },
      { sessionId: 1, fileName: 1, title: 1, createdAt: 1, updatedAt: 1 }
    ).sort({ updatedAt: -1 }).limit(50);
    res.json(sessions);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get("/sessions/:sessionId", async (req, res) => {
  try {
    const session = await Session.findOne({ sessionId: req.params.sessionId });
    if (!session) return res.status(404).json({ error: "Not found" });
    res.json(session);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

app.delete("/sessions/:sessionId", async (req, res) => {
  try {
    await Session.deleteOne({ sessionId: req.params.sessionId });
    res.json({ ok: true });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({ error: "No file uploaded" });

    const sessionId = req.headers["x-session-id"] || Date.now().toString();
    const ext       = path.extname(file.originalname).toLowerCase();
    let content     = "";

    if (ext === ".pdf") {
      const pdfParse = require("pdf-parse");
      content = (await pdfParse(fs.readFileSync(file.path))).text;
    } else if (ext === ".docx") {
      const mammoth = require("mammoth");
      content = (await mammoth.extractRawText({ path: file.path })).value;
    } else if (ext === ".xlsx" || ext === ".xls") {
      const XLSX = require("xlsx");
      const wb   = XLSX.readFile(file.path);
      let text   = "";
      wb.SheetNames.forEach(n => { text += `\n=== ${n} ===\n` + XLSX.utils.sheet_to_csv(wb.Sheets[n]); });
      content = text;
    } else {
      try { content = fs.readFileSync(file.path, "utf8"); } catch { content = "[Could not read file]"; }
    }

    fs.unlinkSync(file.path);

   await Session.findOneAndUpdate(
  { sessionId },
  { fileName: file.originalname, content, history: [], title: file.originalname, updatedAt: new Date() },
  { upsert: true, returnDocument: 'after' } // <--- Use returnDocument: 'after' instead
);

    res.json({ sessionId, fileName: file.originalname });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// All AI routes use streaming — user sees output immediately
app.post("/ask", async (req, res) => {
  const { sessionId, question } = req.body;
  const session = await Session.findOne({ sessionId });
  if (!session) return res.status(404).json({ error: "Session not found." });

  const context = session.content.substring(0, LIMITS.ask);
  const prompt  = `You are a document assistant. Answer based ONLY on the document below. Be concise.\n\nDocument:\n${context}\n\nQuestion: ${question}\nAnswer:`;

  try {
    const fullText = await streamOllama(prompt, res);
    session.history.push({ role: "user", text: question });
    session.history.push({ role: "ai",   text: fullText });
    if (session.history.length <= 2) session.title = question.substring(0, 60);
    session.updatedAt = new Date();
    await session.save();
    res.end();
  } catch (err) {
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

app.post("/summary", async (req, res) => {
  const { sessionId } = req.body;
  const session = await Session.findOne({ sessionId });
  if (!session) return res.status(404).json({ error: "Session not found." });

  const context = session.content.substring(0, LIMITS.summary);
  const prompt  = `Summarize this document in exactly 5 concise bullet points. Each point max 20 words.\n\nDocument:\n${context}\n\nBullet points:`;

  try {
    const fullText = await streamOllama(prompt, res);
    session.history.push({ role: "user", text: "Summarize this document" });
    session.history.push({ role: "ai",   text: fullText });
    session.updatedAt = new Date();
    await session.save();
    res.end();
  } catch (err) { res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`); res.end(); }
});

app.post("/quiz", async (req, res) => {
  const { sessionId } = req.body;
  const session = await Session.findOne({ sessionId });
  if (!session) return res.status(404).json({ error: "Session not found." });

  const context = session.content.substring(0, LIMITS.quiz);
  const prompt  = `Create a 5-question multiple choice quiz from this document. Format:\nQ1. [question]\nA) B) C) D)\nAnswer: [letter]\n\nDocument:\n${context}\n\nQuiz:`;

  try {
    const fullText = await streamOllama(prompt, res);
    session.history.push({ role: "user", text: "Generate a quiz" });
    session.history.push({ role: "ai",   text: fullText });
    session.updatedAt = new Date();
    await session.save();
    res.end();
  } catch (err) { res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`); res.end(); }
});

app.post("/flashcards", async (req, res) => {
  const { sessionId } = req.body;
  const session = await Session.findOne({ sessionId });
  if (!session) return res.status(404).json({ error: "Session not found." });

  const context = session.content.substring(0, LIMITS.flashcards);
  const prompt  = `Create 6 flashcards from this document. Use exactly:\nFront: [question]\nBack: [answer]\n\nDocument:\n${context}\n\nFlashcards:`;

  try {
    const fullText = await streamOllama(prompt, res);
    session.history.push({ role: "user", text: "Generate flashcards" });
    session.history.push({ role: "ai",   text: fullText });
    session.updatedAt = new Date();
    await session.save();
    res.end();
  } catch (err) { res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`); res.end(); }
});

app.post("/clear", async (req, res) => {
  const { sessionId } = req.body;
  await Session.findOneAndUpdate({ sessionId }, { history: [] });
  res.json({ ok: true });
});

app.listen(PORT, async () => {
  console.log(`\n✅ Document Manager running at http://localhost:${PORT}`);
  console.log(`   Model: ${MODEL_NAME} | Streaming: ON\n`);
  try { const { default: open } = await import("open"); open("http://localhost:" + PORT); } catch {}
});