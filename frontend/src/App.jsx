import { useEffect, useRef, useState } from "react";

const API = "http://127.0.0.1:8000/api";
const DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
const PALETTE = [
  { accent:"#6366f1", bg:"rgba(99,102,241,0.08)", border:"rgba(99,102,241,0.35)" },
  { accent:"#0ea5e9", bg:"rgba(14,165,233,0.08)", border:"rgba(14,165,233,0.35)" },
  { accent:"#f59e0b", bg:"rgba(245,158,11,0.08)", border:"rgba(245,158,11,0.35)" },
  { accent:"#22c55e", bg:"rgba(34,197,94,0.08)", border:"rgba(34,197,94,0.35)" },
  { accent:"#a855f7", bg:"rgba(168,85,247,0.08)", border:"rgba(168,85,247,0.35)" },
  { accent:"#f43f5e", bg:"rgba(244,63,94,0.08)", border:"rgba(244,63,94,0.35)" },
];
const cMap = {}; let cIdx = 0;
const getColor = s => { if (!cMap[s]) cMap[s] = PALETTE[cIdx++ % PALETTE.length]; return cMap[s]; };

const safeFetch = async (url, opts = {}) => {
  const res = await fetch(url, opts);
  const txt = await res.text();
  let d; try { d = JSON.parse(txt); } catch { throw new Error("Invalid server response"); }
  if (!res.ok) throw new Error(d.error || `Error ${res.status}`);
  return d;
};

function Spinner({ size = 14 }) {
  return <span style={{ display:"inline-block", width:size, height:size, border:"1.5px solid #333", borderTopColor:"#888", borderRadius:"50%", animation:"spin .7s linear infinite", flexShrink:0 }} />;
}

function Toast({ toasts, remove }) {
  return (
    <div style={{ position:"fixed", bottom:24, right:24, zIndex:9999, display:"flex", flexDirection:"column", gap:8, pointerEvents:"none" }}>
      {toasts.map(t => (
        <div key={t.id} onClick={() => remove(t.id)} style={{
          background:"#111", border:`1px solid ${t.type==="success"?"#22c55e":t.type==="error"?"#f43f5e":"#333"}`,
          borderRadius:10, padding:"11px 15px", maxWidth:320, pointerEvents:"all",
          cursor:"pointer", display:"flex", gap:10, alignItems:"flex-start",
          animation:"fadeUp .22s ease", boxShadow:"0 8px 32px rgba(0,0,0,0.5)"
        }}>
          <span style={{ fontSize:12, color:t.type==="success"?"#22c55e":t.type==="error"?"#f43f5e":"#888", lineHeight:"18px", flexShrink:0, fontWeight:700 }}>
            {t.type==="success"?"✓":t.type==="error"?"✕":"●"}
          </span>
          <div>
            <div style={{ fontWeight:600, fontSize:13, color:"#fff", marginBottom:2 }}>{t.title}</div>
            <div style={{ fontSize:12, color:"#666", lineHeight:1.5 }}>{t.message}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [events, setEvents] = useState([]);
  const [email, setEmail] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [reminder, setReminder] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [reminderLoading, setReminderLoading] = useState(false);
  const [savedTimetables, setSavedTimetables] = useState([]);
  const [activeTimetableId, setActiveTimetableId] = useState(null);
  const [currentTitle, setCurrentTitle] = useState("");
  const [toasts, setToasts] = useState([]);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef(null);
  const tid = useRef(0);

  const addToast = (title, message, type = "info") => {
    const id = tid.current++;
    setToasts(p => [...p, { id, title, message, type }]);
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 5000);
  };

  function notify(title, body) {
    if ("Notification" in window && Notification.permission === "granted") new Notification(title, { body });
  }

  async function fetchSaved(e) {
    if (!e) return;
    try {
      const d = await safeFetch(`${API}/list-timetables/?email=${encodeURIComponent(e)}`);
      setSavedTimetables(d.timetables || []);
      const active = (d.timetables || []).find(t => t.is_active);
      setActiveTimetableId(active ? active.id : null);
    } catch (e) { console.error(e); }
  }

  useEffect(() => {
    if (email.trim()) fetchSaved(email.trim());
    else { setSavedTimetables([]); setActiveTimetableId(null); }
  }, [email]);

  useEffect(() => {
    if (!email.trim()) return;
    if ("Notification" in window && Notification.permission === "default")
      Notification.requestPermission().then(p => { if (p === "granted") addToast("Notifications enabled", "You'll get class reminders.", "success"); });
    const iv = setInterval(async () => {
      try {
        const d = await safeFetch(`${API}/due-reminders/?email=${encodeURIComponent(email.trim())}`);
        (d.reminders || []).forEach(r => { addToast("Class reminder", r.message, "info"); notify(`${r.subject} — 30 min`, r.message); });
      } catch(e) {}
    }, 30000);
    return () => clearInterval(iv);
  }, [email]);

  async function processFile(file) {
    if (!file) return;
    if (!email.trim()) { addToast("Email required", "Enter your email first.", "error"); return; }
    setLoading(true); setSelected(null); setReminder(null);
    const fd = new FormData();
    fd.append("file", file); fd.append("email", email.trim());
    fd.append("title", title.trim()); fd.append("mode", "save_new");
    try {
      const d = await safeFetch(`${API}/parse-schedule/`, { method:"POST", body:fd });
      setEvents(d.events || []); setUploaded(true);
      setCurrentTitle(d.title || title || "My Timetable");
      if (d.timetable_id) setActiveTimetableId(d.timetable_id);
      await fetchSaved(email.trim());
      addToast("Timetable saved", `${d.events?.length || 0} classes found. Reminders scheduled.`, "success");
      notify("HumSabaq AI", `${d.events?.length || 0} classes loaded.`);
    } catch(e) { addToast("Upload failed", e.message, "error"); }
    finally { setLoading(false); if (fileRef.current) fileRef.current.value = ""; }
  }

  async function handleSetActive(id) {
    try {
      await safeFetch(`${API}/set-active-timetable/`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ timetable_id:id, email:email.trim() }) });
      setActiveTimetableId(id); await fetchSaved(email.trim());
      addToast("Activated", "Reminders rescheduled.", "success");
    } catch(e) { addToast("Error", e.message, "error"); }
  }

  async function handleDelete(id) {
    if (!window.confirm("Delete this timetable?")) return;
    try {
      await safeFetch(`${API}/delete-timetable/`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ timetable_id:id, email:email.trim() }) });
      if (activeTimetableId === id) { setActiveTimetableId(null); setEvents([]); setUploaded(false); setCurrentTitle(""); }
      await fetchSaved(email.trim());
      addToast("Deleted", "Timetable removed.", "success");
    } catch(e) { addToast("Error", e.message, "error"); }
  }

  async function handleSelectClass(ev) {
    setSelected(ev); setReminder(null); setReminderLoading(true);
    try {
      const d = await safeFetch(`${API}/get-reminders/`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ subject:ev.subject, day:ev.day, time:ev.time }) });
      setReminder(d);
    } catch(e) {}
    finally { setReminderLoading(false); }
  }

  const byDay = DAYS.reduce((a, d) => { a[d] = events.filter(e => e.day === d); return a; }, {});

  const S = {
    page: { minHeight:"100vh", background:"#0a0a0a", color:"#fff", fontFamily:"'Inter', system-ui, sans-serif", WebkitFontSmoothing:"antialiased" },
    nav: { background:"rgba(10,10,10,0.95)", backdropFilter:"blur(12px)", borderBottom:"1px solid #222", padding:"0 48px", height:60, display:"flex", alignItems:"center", justifyContent:"space-between", position:"sticky", top:0, zIndex:100 },
    navBrand: { display:"flex", alignItems:"center", gap:10 },
    navLogo: { width:30, height:30, background:"#fff", borderRadius:7, display:"flex", alignItems:"center", justifyContent:"center" },
    navName: { fontSize:15, fontWeight:700, color:"#fff", letterSpacing:"-0.3px" },
    navRight: { display:"flex", alignItems:"center", gap:8, fontSize:12, color:"#666" },
    liveDot: { width:6, height:6, borderRadius:"50%", background:"#22c55e", flexShrink:0 },
    main: { maxWidth:1160, margin:"0 auto", padding:"40px 24px 80px" },
    card: { background:"#111", border:"1px solid #222", borderRadius:14, padding:"22px 24px", marginBottom:20 },
    label: { fontSize:11, fontWeight:600, color:"#555", textTransform:"uppercase", letterSpacing:"0.09em", display:"block", marginBottom:7 },
    input: { width:"100%", padding:"11px 13px", background:"#0a0a0a", border:"1px solid #222", borderRadius:9, color:"#fff", fontSize:14, outline:"none", boxSizing:"border-box", fontFamily:"inherit", transition:"border-color 0.15s" },
    sectionLabel: { fontSize:11, fontWeight:600, color:"#444", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:16 },
    pill: { fontSize:10, color:"#22c55e", background:"rgba(34,197,94,0.1)", border:"1px solid rgba(34,197,94,0.25)", borderRadius:20, padding:"2px 9px", fontWeight:700, letterSpacing:"1px" },
    outlineBtn: { padding:"6px 14px", background:"transparent", border:"1px solid #2a2a2a", borderRadius:8, color:"#666", cursor:"pointer", fontSize:12, fontWeight:600, fontFamily:"inherit", transition:"all 0.15s" },
    dangerBtn: { padding:"6px 14px", background:"transparent", border:"1px solid rgba(244,63,94,0.3)", borderRadius:8, color:"#f43f5e", cursor:"pointer", fontSize:12, fontWeight:600, fontFamily:"inherit", transition:"all 0.15s" },
    tag: { fontSize:11, color:"#555", background:"#1a1a1a", border:"1px solid #2a2a2a", borderRadius:6, padding:"3px 10px", fontWeight:500 },
  };

  return (
    <div style={S.page}>
      <Toast toasts={toasts} remove={id => setToasts(p => p.filter(t => t.id !== id))} />

      {/* Navbar */}
      <nav style={S.nav}>
        <div style={S.navBrand}>
          <div style={S.navLogo}><svg viewBox="0 0 16 16" width="16" height="16" style={{fill:"#0a0a0a"}}><path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1Z"/></svg></div>
          <span style={S.navName}>HumSabaq AI</span>
          <span style={{ marginLeft:6, background:"#1a1a1a", border:"1px solid #2a2a2a", borderRadius:20, padding:"2px 10px", fontSize:10, color:"#555", letterSpacing:"1.5px", fontWeight:600 }}>TIMETABLE</span>
        </div>
        <div style={S.navRight}><span style={S.liveDot} />System online</div>
      </nav>

      <div style={S.main}>

        {/* Hero */}
        <div style={{ textAlign:"center", marginBottom:40 }}>
          <div style={{ display:"inline-flex", alignItems:"center", gap:6, background:"#111", border:"1px solid #222", borderRadius:20, padding:"5px 13px", marginBottom:18, fontSize:11, color:"#666", letterSpacing:"0.05em" }}>
            <span style={S.liveDot} /> AI-POWERED SCHEDULE ASSISTANT
          </div>
          <h1 style={{ fontSize:"clamp(32px,5vw,48px)", fontWeight:700, letterSpacing:"-0.04em", color:"#fff", lineHeight:1.1, margin:"0 0 10px" }}>
            Never miss a class again.
          </h1>
          <p style={{ color:"#555", fontSize:15, margin:0, maxWidth:400, marginInline:"auto" }}>
            Upload your timetable. AI reads it, saves it, reminds you 30 min before every class.
          </p>
        </div>

        {/* Upload panel */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:20 }}>
          {/* Inputs */}
          <div style={S.card}>
            <p style={S.sectionLabel}>Setup</p>
            <div style={{ marginBottom:14 }}>
              <label style={S.label}>Your email</label>
              <input type="email" placeholder="student@university.edu" value={email} onChange={e => setEmail(e.target.value)}
                style={S.input}
                onFocus={e => e.target.style.borderColor="#444"}
                onBlur={e => e.target.style.borderColor="#222"} />
            </div>
            <div>
              <label style={S.label}>Timetable name <span style={{ color:"#333", fontWeight:400, textTransform:"none", letterSpacing:0 }}>(optional)</span></label>
              <input type="text" placeholder="e.g. Semester 5 — Spring 2025" value={title} onChange={e => setTitle(e.target.value)}
                style={S.input}
                onFocus={e => e.target.style.borderColor="#444"}
                onBlur={e => e.target.style.borderColor="#222"} />
            </div>
          </div>

          {/* Dropzone */}
          <div
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); processFile(e.dataTransfer.files[0]); }}
            style={{ background: drag ? "rgba(255,255,255,0.03)" : uploaded ? "rgba(34,197,94,0.04)" : "#111",
              border:`2px dashed ${drag ? "#444" : uploaded ? "#22c55e" : "#222"}`,
              borderRadius:14, display:"flex", flexDirection:"column", alignItems:"center",
              justifyContent:"center", cursor:"pointer", padding:"32px 20px", minHeight:140,
              transition:"all 0.2s" }}>
            <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.pdf" style={{ display:"none" }} onChange={e => processFile(e.target.files[0])} />
            {loading ? (
              <><Spinner size={24} /><p style={{ color:"#888", fontWeight:600, fontSize:13, margin:"12px 0 4px" }}>Analysing with AI…</p><p style={{ color:"#333", fontSize:11, margin:0 }}>Reading your timetable</p></>
            ) : uploaded ? (
              <><div style={{ fontSize:22, marginBottom:8, color:"#22c55e" }}>✓</div>
              <p style={{ color:"#22c55e", fontWeight:700, fontSize:14, margin:"0 0 4px", textAlign:"center" }}>{events.length} classes — {currentTitle}</p>
              <p style={{ color:"#333", fontSize:11, margin:0 }}>Click or drop to upload another</p></>
            ) : (
              <><div style={{ width:44, height:44, borderRadius:10, background:"#1a1a1a", display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, marginBottom:12 }}>📤</div>
              <p style={{ color:"#888", fontWeight:600, fontSize:14, margin:"0 0 4px" }}>Drop your timetable here</p>
              <p style={{ color:"#333", fontSize:12, margin:0 }}>JPG, PNG or PDF · Click to browse</p></>
            )}
          </div>
        </div>

        {/* Saved timetables */}
        {savedTimetables.length > 0 && (
          <div style={S.card}>
            <p style={S.sectionLabel}>Saved timetables</p>
            <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
              {savedTimetables.map(tt => (
                <div key={tt.id} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"11px 14px",
                  background: tt.id === activeTimetableId ? "rgba(255,255,255,0.04)" : "#0a0a0a",
                  border:`1px solid ${tt.id === activeTimetableId ? "#333" : "#1e1e1e"}`,
                  borderRadius:9, transition:"all 0.15s" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:12 }}>
                    <div style={{ width:32, height:32, borderRadius:8, background:"#1a1a1a", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14 }}>📅</div>
                    <div>
                      <p style={{ fontSize:13, fontWeight:600, color:"#fff", margin:"0 0 2px" }}>{tt.title}</p>
                      <p style={{ fontSize:11, color:"#444", margin:0 }}>{tt.event_count} classes · {tt.created_at}</p>
                    </div>
                    {tt.id === activeTimetableId && <span style={S.pill}>ACTIVE</span>}
                  </div>
                  <div style={{ display:"flex", gap:7 }}>
                    {tt.id !== activeTimetableId && <button style={S.outlineBtn} onClick={() => handleSetActive(tt.id)} onMouseEnter={e => e.target.style.background="#1a1a1a"} onMouseLeave={e => e.target.style.background="transparent"}>Activate</button>}
                    <button style={S.dangerBtn} onClick={() => handleDelete(tt.id)}>Delete</button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        {events.length > 0 && (
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, marginBottom:20 }}>
            {[
              { icon:"📚", label:"Total classes", value:events.length, color:"#6366f1" },
              { icon:"📅", label:"Days covered", value:DAYS.filter(d => byDay[d].length > 0).length, color:"#0ea5e9" },
              { icon:"🔔", label:"Reminders", value:"Active", color:"#22c55e" },
            ].map((s, i) => (
              <div key={i} style={{ ...S.card, marginBottom:0, padding:"18px 20px" }}>
                <div style={{ display:"flex", alignItems:"center", gap:9, marginBottom:10 }}>
                  <span style={{ fontSize:16 }}>{s.icon}</span>
                  <span style={{ fontSize:10, color:"#444", fontWeight:600, textTransform:"uppercase", letterSpacing:"0.1em" }}>{s.label}</span>
                </div>
                <div style={{ fontSize:26, fontWeight:800, color:s.color, letterSpacing:"-0.5px" }}>{s.value}</div>
              </div>
            ))}
          </div>
        )}

        {/* Weekly calendar */}
        {events.length > 0 && (
          <div style={{ ...S.card, marginBottom:20 }}>
            <p style={{ ...S.sectionLabel, marginBottom:18 }}>Weekly schedule <span style={{ color:"#333", fontWeight:400, textTransform:"none", letterSpacing:0, fontSize:10 }}>— click a class for AI study tips</span></p>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(6, 1fr)", gap:10 }}>
              {DAYS.map(day => (
                <div key={day}>
                  <div style={{ textAlign:"center", fontSize:10, color:"#444", fontWeight:700, textTransform:"uppercase", letterSpacing:"1.5px", marginBottom:10, paddingBottom:8, borderBottom:"1px solid #1e1e1e", display:"flex", alignItems:"center", justifyContent:"center", gap:5 }}>
                    {day.slice(0,3)}
                    {byDay[day].length > 0 && <span style={{ width:15, height:15, background:"#1a1a1a", borderRadius:"50%", fontSize:9, color:"#555", display:"inline-flex", alignItems:"center", justifyContent:"center", fontWeight:700 }}>{byDay[day].length}</span>}
                  </div>
                  <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                    {byDay[day].length === 0 && <div style={{ height:44, border:"1px dashed #1a1a1a", borderRadius:7 }} />}
                    {byDay[day].map((ev, i) => {
                      const c = getColor(ev.subject);
                      const sel = selected?.subject === ev.subject && selected?.time === ev.time && selected?.day === ev.day;
                      return (
                        <div key={i} onClick={() => handleSelectClass(ev)}
                          style={{ background: sel ? c.bg : "#0a0a0a", border:`1px solid ${sel ? c.accent : "#1e1e1e"}`,
                            borderLeft:`3px solid ${c.accent}`, borderRadius:"0 8px 8px 0", padding:"9px 9px",
                            cursor:"pointer", transition:"all 0.15s", transform: sel ? "scale(1.02)" : "scale(1)" }}>
                          <p style={{ fontSize:11, fontWeight:700, color: sel ? c.accent : "#888", margin:"0 0 3px", lineHeight:1.3 }}>{ev.subject}</p>
                          <p style={{ fontSize:10, color:"#444", margin:"0 0 2px" }}>{ev.time}</p>
                          {ev.room && <p style={{ fontSize:9, color:"#333", margin:0 }}>{ev.room}</p>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* AI study brief */}
        {selected && (
          <div style={{ ...S.card, marginBottom:0, animation:"fadeUp .2s ease" }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:18 }}>
              <div style={{ display:"flex", alignItems:"center", gap:12 }}>
                <div style={{ width:38, height:38, borderRadius:9, background:"#1a1a1a", display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>🔔</div>
                <div>
                  <p style={{ fontSize:10, color:"#444", fontWeight:600, textTransform:"uppercase", letterSpacing:"0.1em", margin:"0 0 2px" }}>AI Study Brief</p>
                  <p style={{ fontSize:15, fontWeight:700, color:"#fff", margin:0 }}>{selected.subject}</p>
                </div>
              </div>
              <div style={{ display:"flex", gap:7 }}>
                {[selected.day, selected.time, selected.room].filter(Boolean).map((v, i) => <span key={i} style={S.tag}>{v}</span>)}
              </div>
            </div>
            {reminderLoading ? (
              <div style={{ display:"flex", alignItems:"center", gap:10, color:"#444", padding:"8px 0" }}><Spinner /><span style={{ fontSize:13 }}>Generating study brief…</span></div>
            ) : reminder && (
              <>
                <p style={{ color:"#888", fontSize:14, lineHeight:1.7, marginBottom:18, padding:"13px 15px", background:"#0a0a0a", borderRadius:9, border:"1px solid #1e1e1e", borderLeft:"3px solid #333" }}>{reminder.reminder}</p>
                <p style={{ ...S.sectionLabel, marginBottom:12 }}>Study tips</p>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:9 }}>
                  {(reminder.study_tips || []).map((tip, i) => (
                    <div key={i} style={{ padding:"11px 13px", background:"#0a0a0a", border:"1px solid #1e1e1e", borderRadius:9, display:"flex", gap:9, alignItems:"flex-start" }}>
                      <span style={{ width:20, height:20, background:"#1a1a1a", borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:"#555", fontWeight:700, flexShrink:0, marginTop:1 }}>{i+1}</span>
                      <span style={{ fontSize:13, color:"#888", lineHeight:1.55 }}>{tip}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        input::placeholder { color:#333; }
        ::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:#0a0a0a; } ::-webkit-scrollbar-thumb { background:#222; border-radius:3px; }
        body { background:#0a0a0a; }
      `}</style>
    </div>
  );
}