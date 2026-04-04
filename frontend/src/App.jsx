import { useEffect, useRef, useState } from "react";

const API = "http://127.0.0.1:8000/api";
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

const SUBJECT_COLORS = [
  { border: "#a78bfa", bg: "rgba(167,139,250,0.1)", text: "#a78bfa" },
  { border: "#00B8E6", bg: "rgba(0,184,230,0.1)", text: "#00B8E6" },
  { border: "#F5C842", bg: "rgba(245,200,66,0.1)", text: "#F5C842" },
  { border: "#4ade80", bg: "rgba(74,222,128,0.1)", text: "#4ade80" },
  { border: "#f472b6", bg: "rgba(244,114,182,0.1)", text: "#f472b6" },
];

const colorMap = {};
let colorIdx = 0;
const getSubjectColor = (subject) => {
  if (!colorMap[subject]) colorMap[subject] = SUBJECT_COLORS[colorIdx++ % SUBJECT_COLORS.length];
  return colorMap[subject];
};

const safeFetch = async (url, options = {}) => {
  const res = await fetch(url, options);
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error("Server returned invalid response"); }
  if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
  return data;
};

// ── Toast component ────────────────────────────────────────────────────────
function Toast({ toasts, removeToast }) {
  return (
    <div style={{ position: "fixed", bottom: "24px", right: "24px", zIndex: 9999, display: "flex", flexDirection: "column", gap: "10px" }}>
      {toasts.map(t => (
        <div key={t.id} onClick={() => removeToast(t.id)} style={{
          background: t.type === "success" ? "rgba(74,222,128,0.15)" : t.type === "error" ? "rgba(248,113,113,0.15)" : "rgba(167,139,250,0.15)",
          border: `1px solid ${t.type === "success" ? "#4ade80" : t.type === "error" ? "#f87171" : "#a78bfa"}`,
          borderLeft: `3px solid ${t.type === "success" ? "#4ade80" : t.type === "error" ? "#f87171" : "#a78bfa"}`,
          borderRadius: "10px", padding: "12px 16px", maxWidth: "320px",
          color: "#fff", fontSize: "13px", cursor: "pointer",
          animation: "slideIn 0.3s ease",
          backdropFilter: "blur(10px)",
        }}>
          <div style={{ fontWeight: "700", marginBottom: "4px" }}>
            {t.type === "success" ? "✅" : t.type === "error" ? "❌" : "🔔"} {t.title}
          </div>
          <div style={{ color: "#aaa", fontSize: "12px" }}>{t.message}</div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [events, setEvents]               = useState([]);
  const [email, setEmail]                 = useState("");
  const [title, setTitle]                 = useState("");
  const [loading, setLoading]             = useState(false);
  const [selected, setSelected]           = useState(null);
  const [reminder, setReminder]           = useState(null);
  const [uploaded, setUploaded]           = useState(false);
  const [reminderLoading, setReminderLoading] = useState(false);
  const [savedTimetables, setSavedTimetables] = useState([]);
  const [activeTimetableId, setActiveTimetableId] = useState(null);
  const [currentTitle, setCurrentTitle]   = useState("");
  const [toasts, setToasts]               = useState([]);
  const fileRef = useRef(null);
  const toastId = useRef(0);

  // ── Toast helpers ──────────────────────────────────────────────────────
  function addToast(title, message, type = "info") {
    const id = toastId.current++;
    setToasts(prev => [...prev, { id, title, message, type }]);
    setTimeout(() => removeToast(id), 5000);
  }

  function removeToast(id) {
    setToasts(prev => prev.filter(t => t.id !== id));
  }

  // ── Browser notification helper ───────────────────────────────────────
  function showBrowserNotification(title, body) {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(title, { body, requireInteraction: true });
    }
  }

  async function fetchSaved(userEmail) {
    if (!userEmail) return;
    try {
      const data = await safeFetch(`${API}/list-timetables/?email=${encodeURIComponent(userEmail)}`);
      setSavedTimetables(data.timetables || []);
      const active = (data.timetables || []).find(t => t.is_active);
      setActiveTimetableId(active ? active.id : null);
    } catch(e) { console.error(e); }
  }

  useEffect(() => {
    if (email.trim()) fetchSaved(email.trim());
    else { setSavedTimetables([]); setActiveTimetableId(null); }
  }, [email]);

  // ── Request notification permission + poll for reminders ─────────────
  useEffect(() => {
    if (!email.trim()) return;

    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().then(perm => {
        if (perm === "granted") {
          addToast("Notifications Enabled", "You will receive class reminders!", "success");
          showBrowserNotification("📚 Timetable Manager", "Notifications enabled! You will get class reminders.");
        }
      });
    }

    const interval = setInterval(async () => {
      try {
        const data = await safeFetch(`${API}/due-reminders/?email=${encodeURIComponent(email.trim())}`);
        if (data.reminders && data.reminders.length > 0) {
          for (const item of data.reminders) {
            // ✅ In-app toast alert
            addToast(`Class Reminder`, item.message, "reminder");
            // ✅ Browser notification
            showBrowserNotification(`📚 Timetable Manager— ${item.subject}`, item.message);
          }
        }
      } catch(e) { console.error("Reminder polling failed", e); }
    }, 30000);

    return () => clearInterval(interval);
  }, [email]);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!email.trim()) {
      addToast("Email Required", "Please enter your email first.", "error");
      return;
    }

    setLoading(true);
    setSelected(null);
    setReminder(null);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("email", email.trim());
    fd.append("title", title.trim());
    fd.append("mode", "save_new");

    try {
      const data = await safeFetch(`${API}/parse-schedule/`, { method: "POST", body: fd });
      setEvents(data.events || []);
      setUploaded(true);
      setCurrentTitle(data.title || title || "My Timetable");
      if (data.timetable_id) setActiveTimetableId(data.timetable_id);
      await fetchSaved(email.trim());
      addToast("Timetable Saved!", `${data.events?.length || 0} classes found. Reminders scheduled!`, "success");
      showBrowserNotification("📚 Timetable Manager", `Timetable uploaded! ${data.events?.length || 0} classes found.`);
    } catch(e) {
      addToast("Upload Failed", e.message, "error");
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  // ✅ FIX: email now included in request body
  async function handleSetActive(id) {
    try {
      await safeFetch(`${API}/set-active-timetable/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timetable_id: id, email: email.trim() })
      });
      setActiveTimetableId(id);
      await fetchSaved(email.trim());
      addToast("Timetable Activated", "Reminders rescheduled for this timetable.", "success");
    } catch(e) {
      addToast("Error", e.message, "error");
    }
  }

  // ✅ FIX: email now included in request body
  async function handleDelete(id) {
    if (!window.confirm("Delete this timetable?")) return;
    try {
      await safeFetch(`${API}/delete-timetable/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timetable_id: id, email: email.trim() })
      });
      if (activeTimetableId === id) {
        setActiveTimetableId(null);
        setEvents([]);
        setUploaded(false);
        setCurrentTitle("");
      }
      await fetchSaved(email.trim());
      addToast("Deleted", "Timetable removed successfully.", "success");
    } catch(e) {
      addToast("Delete Failed", e.message, "error");
    }
  }

  async function handleSelectClass(ev) {
    setSelected(ev);
    setReminder(null);
    setReminderLoading(true);
    try {
      const data = await safeFetch(`${API}/get-reminders/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject: ev.subject, day: ev.day, time: ev.time })
      });
      setReminder(data);
    } catch(e) { console.error(e); }
    finally { setReminderLoading(false); }
  }

  const byDay = DAYS.reduce((acc, day) => {
    acc[day] = events.filter(e => e.day === day);
    return acc;
  }, {});

  return (
    <div style={{ minHeight: "100vh", background: "#000000", color: "#ffffff", fontFamily: "sans-serif" }}>

      {/* Toast notifications */}
      <Toast toasts={toasts} removeToast={removeToast} />

      {/* Navbar */}
      <div style={{ background: "#111111", borderBottom: "1px solid #333333", padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "22px" }}>📚</span>
          <span style={{ fontWeight: "900", fontSize: "20px" }}>Timetable <span style={{ color: "#ffffff" }}>Manager</span></span>
          <span style={{ background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "20px", padding: "3px 12px", fontSize: "10px", color: "#aaaaaa", letterSpacing: "2px" }}>SCHEDULE ASSISTANT</span>
        </div>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 8px #4ade80" }} />
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "32px 24px" }}>

        {/* Hero */}
        <div style={{ textAlign: "center", marginBottom: "32px" }}>
          <h1 style={{ fontSize: "32px", fontWeight: "900", margin: "0 0 8px", paddingBottom: "6px", lineHeight: "1.2", color: "#ffffff" }}>
            Never Miss a Class Again
          </h1>
          <p style={{ color: "#555", fontSize: "14px", margin: 0 }}>
            Upload your timetable 
          </p>
        </div>

        {/* Upload card */}
        <div style={{ background: "#1a1a1a", border: "1px solid #333333", borderRadius: "16px", padding: "24px", marginBottom: "24px" }}>

          <div style={{ marginBottom: "12px" }}>
            <label style={{ fontSize: "11px", color: "#888888", fontFamily: "monospace", letterSpacing: "1px", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>Your Email</label>
            <input type="email" placeholder="yourname@gmail.com" value={email} onChange={e => setEmail(e.target.value)}
              style={{ width: "100%", padding: "12px 14px", background: "#2a2a2a", border: "1px solid #444444", borderRadius: "8px", color: "#fff", fontSize: "14px", outline: "none", boxSizing: "border-box" }} />
          </div>

          <div style={{ marginBottom: "16px" }}>
            <label style={{ fontSize: "11px", color: "#888888", fontFamily: "monospace", letterSpacing: "1px", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>Timetable Name (optional)</label>
            <input type="text" placeholder="e.g. Semester 1 Timetable" value={title} onChange={e => setTitle(e.target.value)}
              style={{ width: "100%", padding: "12px 14px", background: "#2a2a2a", border: "1px solid #444444", borderRadius: "8px", color: "#fff", fontSize: "14px", outline: "none", boxSizing: "border-box" }} />
          </div>

          <div onClick={() => fileRef.current.click()} style={{ border: `2px dashed ${uploaded ? "#4ade80" : "#444444"}`, borderRadius: "12px", padding: "36px", textAlign: "center", cursor: "pointer", background: uploaded ? "rgba(74,222,128,0.05)" : "transparent", transition: "all 0.3s" }}>
            <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.pdf" style={{ display: "none" }} onChange={handleUpload} />
            <div style={{ fontSize: "32px", marginBottom: "8px" }}>{uploaded ? "✅" : "📸"}</div>
            <p style={{ color: uploaded ? "#4ade80" : "#ffffff", fontWeight: "700", margin: "0 0 4px" }}>
              {uploaded ? `${events.length} classes found — ${currentTitle}` : "Click to upload timetable photo"}
            </p>
            <p style={{ color: "#777777", fontSize: "12px", margin: 0 }}>JPG, PNG or PDF</p>
          </div>

          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "14px", color: "#aaaaaa" }}>
              <div style={{ width: "14px", height: "14px", border: "2px solid #aaaaaa", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
              <span style={{ fontSize: "13px" }}>Reading your timetable with AI...</span>
            </div>
          )}
        </div>

        {/* Saved timetables */}
        {savedTimetables.length > 0 && (
          <div style={{ background: "#1a1a1a", border: "1px solid #333333", borderRadius: "16px", padding: "20px", marginBottom: "24px" }}>
            <p style={{ fontFamily: "monospace", fontSize: "10px", color: "#777777", letterSpacing: "3px", textTransform: "uppercase", marginBottom: "14px" }}>// Saved Timetables</p>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {savedTimetables.map(tt => (
                <div key={tt.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", background: tt.id === activeTimetableId ? "rgba(255,255,255,0.08)" : "#2a2a2a", border: `1px solid ${tt.id === activeTimetableId ? "#888888" : "#444444"}`, borderRadius: "8px" }}>
                  <div>
                    <span style={{ fontSize: "13px", fontWeight: "700", color: "#ffffff" }}>{tt.title}</span>
                    <span style={{ marginLeft: "8px", fontSize: "10px", color: "#888888", fontFamily: "monospace" }}>{tt.event_count} classes · {tt.created_at}</span>
                    {tt.id === activeTimetableId && <span style={{ marginLeft: "8px", fontSize: "10px", color: "#4ade80", fontFamily: "monospace" }}>● ACTIVE</span>}
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    {tt.id !== activeTimetableId && (
                      <button onClick={() => handleSetActive(tt.id)} style={{ padding: "6px 12px", background: "transparent", border: "1px solid #888888", borderRadius: "6px", color: "#ffffff", cursor: "pointer", fontSize: "11px" }}>
                        Set Active
                      </button>
                    )}
                    <button onClick={() => handleDelete(tt.id)} style={{ padding: "6px 12px", background: "transparent", border: "1px solid #f87171", borderRadius: "6px", color: "#f87171", cursor: "pointer", fontSize: "11px" }}>
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        {events.length > 0 && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "12px", marginBottom: "24px" }}>
            {[
              { icon: "📚", label: "Total Classes", value: events.length },
              { icon: "📅", label: "Days Covered", value: DAYS.filter(d => byDay[d].length > 0).length },
              { icon: "📧", label: "Email Alerts", value: "Active" },
            ].map((stat, i) => (
              <div key={i} style={{ background: "#1a1a1a", border: "1px solid #333333", borderRadius: "12px", padding: "18px", textAlign: "center" }}>
                <div style={{ fontSize: "22px", marginBottom: "6px" }}>{stat.icon}</div>
                <div style={{ fontSize: "20px", fontWeight: "900", color: "#ffffff", marginBottom: "4px" }}>{stat.value}</div>
                <div style={{ fontSize: "10px", color: "#777777", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "1px" }}>{stat.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Calendar */}
        {events.length > 0 && (
          <div style={{ background: "#1a1a1a", border: "1px solid #333333", borderRadius: "16px", padding: "24px", marginBottom: "24px" }}>
            <p style={{ fontFamily: "monospace", fontSize: "10px", color: "#777777", letterSpacing: "3px", textTransform: "uppercase", marginBottom: "20px" }}>// Weekly Schedule — Click any class for reminders</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: "10px" }}>
              {DAYS.map(day => (
                <div key={day}>
                  <div style={{ textAlign: "center", fontSize: "10px", color: "#888888", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "10px", paddingBottom: "8px", borderBottom: "1px solid #333333" }}>
                    {day.slice(0,3)}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {byDay[day].length === 0 && <div style={{ height: "40px", border: "1px dashed #333333", borderRadius: "6px" }} />}
                    {byDay[day].map((ev, i) => {
                      const color = getSubjectColor(ev.subject);
                      const isSel = selected?.subject === ev.subject && selected?.time === ev.time && selected?.day === ev.day;
                      return (
                        <div key={i} onClick={() => handleSelectClass(ev)} style={{ borderLeft: `3px solid ${isSel ? "#fff" : color.border}`, background: isSel ? "rgba(255,255,255,0.1)" : color.bg, borderRadius: "6px", padding: "10px 8px", cursor: "pointer", transition: "all 0.2s" }}>
                          <p style={{ fontSize: "11px", fontWeight: "700", color: isSel ? "#fff" : color.text, margin: "0 0 3px" }}>{ev.subject}</p>
                          <p style={{ fontSize: "10px", color: "#888888", margin: "0 0 2px" }}>{ev.time}</p>
                          {ev.room && <p style={{ fontSize: "9px", color: "#666666", margin: 0 }}>{ev.room}</p>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Reminder panel */}
        {selected && (
          <div style={{ background: "#1a1a1a", border: "1px solid #333333", borderLeft: "3px solid #ffffff", borderRadius: "16px", padding: "24px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
              <span style={{ fontSize: "20px" }}>🔔</span>
              <div>
                <p style={{ fontFamily: "monospace", fontSize: "10px", color: "#aaaaaa", letterSpacing: "2px", textTransform: "uppercase", margin: 0 }}>Reminder</p>
                <p style={{ fontSize: "15px", fontWeight: "700", color: "#fff", margin: 0 }}>{selected.subject} · {selected.day} · {selected.time}</p>
              </div>
            </div>

            {reminderLoading ? (
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#aaaaaa" }}>
                <div style={{ width: "14px", height: "14px", border: "2px solid #aaaaaa", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                <span style={{ fontSize: "13px" }}>Getting AI study tips...</span>
              </div>
            ) : reminder && (
              <>
                <p style={{ color: "#cccccc", fontSize: "14px", marginBottom: "20px", lineHeight: "1.6" }}>{reminder.reminder}</p>
                <p style={{ fontFamily: "monospace", fontSize: "10px", color: "#777777", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "12px" }}>AI Study Tips</p>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  {(reminder.study_tips || []).map((tip, i) => (
                    <div key={i} style={{ display: "flex", gap: "12px", alignItems: "flex-start", padding: "12px 16px", background: "#2a2a2a", borderRadius: "8px", border: "1px solid #444444" }}>
                      <span style={{ color: "#aaaaaa", fontSize: "12px", marginTop: "1px" }}>▸</span>
                      <span style={{ color: "#cccccc", fontSize: "13px", lineHeight: "1.5" }}>{tip}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
        input::placeholder { color: #555555; }
        input:focus { border-color: #888888 !important; }
      `}</style>
    </div>
  );
}