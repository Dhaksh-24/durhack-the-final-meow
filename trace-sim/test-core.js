// Tests the model core of index.html in Node: node trace-sim/test-core.js [episodes]
// Checks task 1 (SPEC section 8): with false alarms on, Trace still beats the baseline,
// calibration bins stay within ±10 points, and one false sighting doesn't collapse the belief.
// All results are simulated.
const vm = require("vm"), fs = require("fs"), path = require("path");
const src = fs.readFileSync(path.join(__dirname, "index.html"), "utf8").replace(/\r\n/g, "\n");
const js = src.match(/<script>\n\(function([\s\S]*)<\/script>/)[0].slice(8, -9);
const m = { exports: {} };
vm.runInNewContext(js, { module: m, Math, console, Float64Array, Int16Array, JSON, Set, Object, String, Number, isFinite, Infinity, RegExp, Array });
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

// Messy real streams (robustness): flicker, low-confidence false sightings, unknown zones and items,
// duplicates, shuffled lines, a long pause, items seen on the person, no person events at all.
console.log("\nmessy event streams");
C.setupMap(JSON.parse(JSON.stringify(C.DEFAULT_MAP))); C.setFalseRate(0.1);
const T0 = 1794734600, line = (dt, o) => JSON.stringify(Object.assign({ t: T0 + dt, camera: "top" }, o));
const runEvents = (txt) => {
  const r = C.replayFromEvents(txt), B = {};
  for (const it of C.ITEMS) B[it.id] = Float64Array.from(C.PRIORS[it.id]);
  let bad = 0;
  for (const o of r.ticks) for (const it of C.ITEMS) {
    B[it.id] = C.filterStep(B[it.id], it, o);
    let sum = 0; for (const x of B[it.id]) { if (!(x >= 0) || !isFinite(x)) bad++; sum += x; }
    if (Math.abs(sum - 1) > 1e-9) bad++;
  }
  return { r, B, bad, top: (id) => C.rank(B[id])[0].pl.id };
};
{
  const L = [];
  for (let s = 0; s < 3600; s += 3) {   // an hour: keys flicker on the desk, glasses steady on the shelf
    L.push(line(s, { event: (s / 3) % 2 ? "disappeared" : "appeared", item: "keys-1", zone: "desk", confidence: 0.7 }));
    if (s % 60 === 0) L.push(line(s + 1, { event: "appeared", item: "glasses-1", zone: "shelf", confidence: 0.8 }));
    if (s % 97 === 0) L.push(line(s + 2, { event: "appeared", item: "keys-2", zone: "sofa", confidence: 0.2 }), line(s + 5, { event: "disappeared", item: "keys-2", zone: "sofa", confidence: 0 }));
    if (s % 300 === 0) L.push(line(s, { event: "appeared", item: "remote-1", zone: "garage", confidence: 0.9 }), line(s, { event: "person_here", zone: "hallway" }));
  }
  L.push(line(3601, { event: "appeared", item: "wallet-1", zone: "person", confidence: 0.9 }), line(3602, { event: "appeared", item: "glasses-1", zone: "shelf", confidence: null, time: null }));
  L.push(line(3600 + 3 * 3600, { event: "appeared", item: "keys-1", zone: "desk", confidence: 0.8 }), line(3630 + 3 * 3600, { event: "something_changed", zone: "desk" }));   // after a 3-hour pause, seen for 30 s
  const messy = L.concat(L).sort(() => Math.random() - 0.5).join("\n") + "\nnot json\n{}\n[1,2]\nnull\n";
  const t0 = Date.now(), M = runEvents(messy), ms = Date.now() - t0;
  console.log(`  ${M.r.count} events → ${M.r.ticks.length} ticks in ${ms} ms; notes: ${M.r.problems.join("; ")}`);
  check(M.bad === 0, "no NaN, negative or unnormalised beliefs");
  check(ms < 2000, "an hour of busy events (plus a long pause) replays in under 2 s");
  check(M.r.ticks.length <= 2 * 360 + 5, "a 3-hour pause is shortened to an hour, not 1,080 empty ticks");
  console.log(`  keys: ${C.rank(M.B.keys).slice(0, 3).map((x) => x.pl.id + " " + pct(x.p)).join(", ")}; glasses: ${C.rank(M.B.glasses).slice(0, 2).map((x) => x.pl.id + " " + pct(x.p)).join(", ")}`);
  check(M.top("keys") === "desk" && M.top("glasses") === "shelf", "flickering keys stay on the desk; glasses on the shelf despite unknown zones and items");
  check(C.rank(M.B.wallet)[0].pl.id === "person", "an item seen on the person is with the person");
  check(M.r.problems.some((p) => /garage|hallway/.test(p)) && M.r.problems.some((p) => /remote-1/.test(p)), "unknown zones and items are reported");
}
{   // a false 0.2 sighting of the keys on the sofa every minute while a phone keeps seeing them on the desk
  const L = [line(0, { event: "appeared", item: "keys-1", zone: "desk", confidence: 0.85 })];
  for (let s = 30; s < 1800; s += 60) L.push(line(s, { event: "appeared", item: "keys-2", zone: "sofa", confidence: 0.2 }), line(s + 9, { event: "disappeared", item: "keys-2", zone: "sofa" }));
  const F = runEvents(L.join("\n"));
  check(at(F.B.keys, "desk") > 0.9, `low-confidence false sightings elsewhere don't move the keys (desk ${pct(at(F.B.keys, "desk"))})`);
}
{   // timestamps: ms instead of s, null t, lines with only a clock time among unix lines
  const base = [line(0, { time: "09:23:20", event: "person_here", zone: "sofa" }), line(30, { time: "09:23:50", event: "appeared", item: "keys-1", zone: "sofa", confidence: 0.9 }), line(300, { time: "09:28:20", event: "person_left", zone: "sofa" })];
  const ref = C.replayFromEvents(base.join("\n")).ticks.length;
  const ms = base.map((l) => { const e = JSON.parse(l); e.t *= 1000; return JSON.stringify(e); });
  const mixed = base.map((l, k) => { const e = JSON.parse(l); if (k === 1) e.t = null; if (k === 2) delete e.t; return JSON.stringify(e); });
  check(ref === 31 && C.replayFromEvents(ms.join("\n")).ticks.length === ref, "t in milliseconds is read as seconds");
  const R = C.replayFromEvents(mixed.join("\n"));
  check(R.ticks.length === ref && R.ticks[3].det.keys === "sofa", "lines with only a clock time line up with unix-time lines (t: null isn't 1970)");
  const nc = C.replayFromEvents(line(0, { event: "appeared", item: "keys-1", zone: "desk", confidence: null }));
  check(nc.ticks[0].conf.keys === 0.8, "confidence: null counts as 0.8, not 0");
}
// The stage demo (HANDOVER 5): presenter at the sofa, keys seen, slid under a cloth while they sit
// there, then they walk to the desk. With and without person events (multicam --no-people, or none).
console.log("\nstage demo: keys hidden under a cloth on a watched zone");
const DEMO = [line(0, { event: "person_here", zone: "sofa" }), line(5, { event: "appeared", item: "keys-1", zone: "sofa", confidence: 0.85 }),
  line(60, { event: "disappeared", item: "keys-1", zone: "sofa", confidence: 0 }), line(90, { event: "person_left", zone: "sofa" }),
  line(95, { event: "person_here", zone: "desk" }), line(155, { event: "something_changed", zone: "desk" })];
for (const [name, txt, min] of [["with person events", DEMO.join("\n"), 0.75],
    ["no person events", DEMO.filter((l) => !/person_/.test(l)).join("\n"), 0.6],
    ["people detector missed them hiding the keys", [line(0, { event: "person_here", zone: "desk" }), line(1, { event: "person_left", zone: "desk" })].concat(DEMO.slice(1)).join("\n"), 0.6]]) {
  const D = runEvents(txt), b = D.B.keys, plan = C.rankAfter(b);
  const nb = C.failUpdate(b, plan[0].pl), next = C.rankAfter(nb, plan[0].pl.name)[0];
  console.log(`  ${name}: ${plan.slice(0, 3).map((x) => x.pl.id + " " + pct(x.p)).join(", ")}; after 'not there' in pockets: ${next.pl.id} ${pct(next.p)} (${C.probWord(next.p)})`);
  check(D.bad === 0 && plan[0].pl.id === "person" && next.pl.id === "sofa" && next.p >= min, name + ": pockets first, then under the sofa (≥ " + pct(min) + ")");
}

// The guide loop: checks with confidence, the trail, and SPEC 5.2 validation of LLM answers.
console.log("\nguide: checks, trail, validation");
C.setupMap(JSON.parse(JSON.stringify(C.DEFAULT_MAP)));
const pl = (id) => C.PLACES.find((x) => x.id === id);
const b0 = Float64Array.from(C.PRIORS.keys), sofaP = (b) => C.placeP(b, pl("sofa"));
const full = C.checkUpdate(b0, pl("sofa"), "all", 1), half = C.checkUpdate(b0, pl("sofa"), "all", 0.5), same = C.failUpdate(b0, pl("sofa"));
check(Math.abs(sofaP(full) - sofaP(same)) < 1e-12, "a sure check equals the old 'not there'");
check(sofaP(half) > sofaP(full) && sofaP(half) < sofaP(b0), "'I think I looked' lowers it less");
const under = C.checkUpdate(b0, pl("sofa"), "covered", 1), i = pl("sofa").i;
check(under[2 * i] / under[2 * i + 1] > b0[2 * i] / b0[2 * i + 1], "looking under things only lowers the covered part");
check(["very likely", "probably", "might be", "possibly"].join() === [0.9, 0.7, 0.5, 0.1].map(C.probWord).join(), "probability words come from code");
const log = [{ s: 0, zone: "desk" }, { s: 60, zone: "desk" }, { s: 120, zone: "sofa" }, { s: 180, zone: null }, { s: 240, zone: "outside" }];
const vis = C.trailVisits(log, 60, 300);
check(JSON.stringify(vis) === JSON.stringify([{ zone: "desk", from: 60, to: 120 }, { zone: "sofa", from: 120, to: 180 }, { zone: "outside", from: 240, to: 300 }]), "trail since last seen: desk 1 min, sofa 1 min, outside 1 min");
// a belief where the desk is clearly the best place
let bd = Float64Array.from(b0); for (let k = 0; k < 3; k++) bd = C.filterStep(bd, C.ITEMS[0], { prev: "kitchen", person: "kitchen", det: { keys: "desk" }, conf: { keys: 0.9 } });
const dw = C.probWord(C.placeP(bd, pl("desk"))), facts = "last seen 12 minutes ago at 14:05";
const v = (say, places) => C.validateSay(say, places, bd, ["Sofa"], facts);
check(v("Not on the sofa. The desk is " + dw + " the place to check next.", [{ zone: "desk", label: dw }]).ok, "good answer passes ('" + dw + "' for the desk)");
check(!v("They are definitely on the desk.").ok, "rejects 'definitely'");
check(!v("It's on the desk.").ok, "rejects stating where it is as fact");
check(!v("Check the desk, " + (dw === "possibly" ? "very likely" : "possibly") + ".").ok, "rejects a probability word that doesn't match the plan");
check(!v("Check the desk: 87% likely.").ok, "rejects percentages");
check(!v("Your husband moved them to the desk.").ok, "rejects saying who moved it");
check(!v("You had them on the desk about 3 minutes ago.").ok && v("You had them on the desk 12 minutes ago.").ok, "times only from the model");
check(!v("Check the desk.", [{ zone: "garage", label: "probably" }]).ok, "rejects places that aren't in the plan");

console.log(ok ? "\nall checks passed" : "\nSOME CHECKS FAILED");
process.exit(ok ? 0 : 1);
