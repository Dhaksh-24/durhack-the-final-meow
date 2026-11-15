// Tests the model core of index.html in Node: node trace-sim/test-core.js [episodes]
// Checks task 1 (SPEC section 8): with false alarms on, Trace still beats the baseline,
// calibration bins stay within ±10 points, and one false sighting doesn't collapse the belief.
// All results are simulated.
const vm = require("vm"), fs = require("fs"), path = require("path");
const src = fs.readFileSync(path.join(__dirname, "index.html"), "utf8").replace(/\r\n/g, "\n");
const js = src.match(/<script>\n\(function([\s\S]*)<\/script>/)[0].slice(8, -9);
const m = { exports: {} };
vm.runInNewContext(js, { module: m, Math, console, Float64Array, JSON, Set, Object, String, isFinite, Infinity, RegExp, Array });
const C = m.exports;
C.setupMap(JSON.parse(JSON.stringify(C.DEFAULT_MAP)));
const N = +process.argv[2] || 1000;
let ok = true;
const check = (cond, msg) => { console.log((cond ? "  PASS " : "  FAIL ") + msg); if (!cond) ok = false; };
const pct = (x) => (100 * x).toFixed(0) + "%";

function summary(r, k) {
  const a = r.map((x) => x[k]);
  return { first: a.filter((x) => x.found && x.n === 1).length / r.length,
           within3: a.filter((x) => x.found && x.n <= 3).length / r.length,
           checks: a.reduce((p, x) => p + x.n, 0) / r.length,
           secs: a.reduce((p, x) => p + x.t, 0) / r.length };
}

for (const f of [0, 0.05, 0.1, 0.3]) {
  C.setFalseRate(f);
  const r = Array.from({ length: N }, () => C.runEpisode());
  const T = summary(r, "trace"), B = summary(r, "base");
  console.log(`\nfalse-alarm rate ${f} per watched zone per tick (${N} simulated days)`);
  console.log(`  Trace    first ${pct(T.first)}  ≤3 ${pct(T.within3)}  checks ${T.checks.toFixed(2)}  ${T.secs.toFixed(0)} s`);
  console.log(`  Baseline first ${pct(B.first)}  ≤3 ${pct(B.within3)}  checks ${B.checks.toFixed(2)}  ${B.secs.toFixed(0)} s`);
  check(T.first > B.first && T.secs < B.secs, "Trace beats the baseline");
  const edges = [0, 0.2, 0.4, 0.6, 0.8, 1.01];
  let worst = 0;
  for (let i = 0; i < 5; i++) {
    const b = r.filter((x) => x.topP >= edges[i] && x.topP < edges[i + 1]);
    if (b.length < 100) continue;   // too few to judge (±5 points of sampling noise at n=100)
    const p = b.reduce((s, x) => s + x.topP, 0) / b.length, h = b.filter((x) => x.topHit).length / b.length;
    // off by more than 10 points beyond sampling noise (2 standard errors of the hit rate)
    worst = Math.max(worst, Math.abs(p - h) - 2 * Math.sqrt(h * (1 - h) / b.length));
    console.log(`  bin ${edges[i].toFixed(1)}–${Math.min(1, edges[i + 1]).toFixed(1)}: n=${b.length}  said ${pct(p)}  hit ${pct(h)}`);
  }
  check(worst <= 0.10, `calibration within ±10 points, allowing for sampling noise (worst excess ${(100 * worst).toFixed(1)})`);
}

// One false sighting: keys believed to be on the desk, then a single sighting on the sofa.
console.log("\nsingle sighting on the sofa while the keys are believed to be on the desk");
C.setFalseRate(0.1);
const keys = C.ITEMS.find((it) => it.id === "keys");
let b = Float64Array.from(C.PRIORS.keys);
const quiet = (det) => ({ prev: "kitchen", person: "kitchen", det: { keys: det }, conf: { keys: det ? 0.9 : null } });
for (let t = 0; t < 5; t++) b = C.filterStep(b, keys, quiet("desk"));
const zi = (id) => C.DEFAULT_MAP.zones.findIndex((z) => z.id === id);
const at = (b, id) => b[2 * zi(id)] + b[2 * zi(id) + 1];
console.log(`  before: desk ${pct(at(b, "desk"))}, sofa ${pct(at(b, "sofa"))}`);
for (const c of [0.1, 0.3, 0.5, 0.9]) {
  const nb = C.filterStep(b, keys, { prev: "kitchen", person: "kitchen", det: { keys: "sofa" }, conf: { keys: c } });
  console.log(`  sighting at confidence ${c}: desk ${pct(at(nb, "desk"))}, sofa ${pct(at(nb, "sofa"))}`);
  if (c <= 0.3) check(at(nb, "desk") > 0.5, `a ${c} sighting doesn't collapse the belief`);
  if (c === 0.9) check(at(nb, "sofa") > 0.5, "a 0.9 sighting moves the belief a lot");
}

// Task 3a: replay a recorded events.jsonl (sample-events.jsonl is a scripted demo session).
console.log("\nreplaying sample-events.jsonl");
const text = fs.readFileSync(path.join(__dirname, "sample-events.jsonl"), "utf8");
const replayed = (txt) => {
  const r = C.replayFromEvents(txt), B = {};
  for (const it of C.ITEMS) B[it.id] = Float64Array.from(C.PRIORS[it.id]);
  const trail = [];
  r.ticks.forEach((o) => { for (const it of C.ITEMS) B[it.id] = C.filterStep(B[it.id], it, o); trail.push(C.rank(B.keys)[0].pl.id); });
  return { r, B, trail };
};
const A = replayed(text);
console.log(`  ${A.r.count} events → ${A.r.ticks.length} ticks; notes: ${A.r.problems.join("; ") || "none"}`);
check(A.r.ticks.length === 91, "one tick per 10 s from the first event to the last");
check(A.trail[3] === "desk" && at(A.B.wallet, "desk") > 0.9, "a sighting puts the keys and wallet on the desk");
const topEnd = C.rank(A.B.keys)[0];
console.log(`  keys at the end: ${C.rank(A.B.keys).slice(0, 3).map((x) => x.pl.id + " " + pct(x.p)).join(", ")}`);
check(topEnd.pl.id === "hook", "after being carried to the unwatched coat hook, the hook is the top guess");
check(A.r.problems.some((p) => /remote-1/.test(p)), "items Trace doesn't track are reported, not crashed on");
const shuffled = text.trim().split("\n").reverse().join("\n");
const S = replayed(shuffled);
check(JSON.stringify(S.trail) === JSON.stringify(A.trail), "events are replayed in time order whatever the file order");
const bad = C.replayFromEvents('not json\n{"time":"10:00:00","event":"appeared","item":"keys-1","zone":"garage","confidence":0.9}\n');
check(bad.ticks.length === 1 && bad.problems.length === 2, "bad lines and unknown zones are reported: " + bad.problems.join("; "));

// Task 3b: live events arrive one at a time; each burst becomes a tick at once.
console.log("\nlive feed (the same recording, one event at a time)");
const evs = text.trim().split("\n").map((l) => JSON.parse(l));
const runLive = (batches) => {
  const f = C.liveFeed(), B = {};
  for (const it of C.ITEMS) B[it.id] = Float64Array.from(C.PRIORS[it.id]);
  let n = 0;
  for (const batch of batches) for (const o of f.push(batch)) { n++; for (const it of C.ITEMS) B[it.id] = C.filterStep(B[it.id], it, o); }
  return { f, B, n };
};
const L1 = runLive(evs.map((e) => [e])), L2 = runLive([evs]);
console.log(`  ${L1.f.count} events → ${L1.n} ticks; keys: ${C.rank(L1.B.keys).slice(0, 3).map((x) => x.pl.id + " " + pct(x.p)).join(", ")}`);
check(L1.n === L2.n && JSON.stringify(C.rank(L1.B.keys).map((x) => x.pl.id)) === JSON.stringify(C.rank(L2.B.keys).map((x) => x.pl.id)),
  "one event at a time gives the same result as the whole batch");
check(C.rank(L1.B.keys)[0].pl.id === "hook", "live ends on the same top guess as the replay (coat hook)");
const f3 = C.liveFeed(); f3.push([evs[1]]);
check(f3.idle(evs[1].t + 35).length === 3, "35 s of silence adds 3 idle ticks");
check(f3.push([{ t: evs[1].t + 36, event: "person_here", zone: "desk" }]).length === 1, "a new event is applied at once, not at the next 10 s step");

console.log(ok ? "\nall checks passed" : "\nSOME CHECKS FAILED");
process.exit(ok ? 0 : 1);
