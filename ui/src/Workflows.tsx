// The ComfyUI side of the picture steps: which model draws, the words added to a description, and Edit picture.
import { useEffect, useId, useState, type ReactNode } from "react";
import { api, fileUrl, type Meta, type Workflow } from "./api";
import { useBusy } from "./hooks";

type Chain = (asset: string, parent: string | null, steps: [string, object][]) => Promise<string | null>;

let asked: Promise<Workflow[]> | null = null; // once per page load: the list changes with Setup, not with a job
let framings: Promise<Record<string, string>> | null = null;

/** State kept in this browser (the user's own defaults); `null` removes it. */
function useStored<T>(key: string, start: T): [T, (v: T) => void] {
  const read = (): T => {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? start : (JSON.parse(raw) as T);
    } catch {
      return start;
    }
  };
  const [value, setValue] = useState<T>(read);
  useEffect(() => setValue(read()), [key]);
  return [
    value,
    (v: T) => {
      if (v === null) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(v));
      setValue(v);
    },
  ];
}

/** [the chosen workflow's id ("" = the built-in model), its picker]: the picker is null when there is nothing to choose
 * (no ComfyUI, or no workflow of this kind), so a page without ComfyUI looks as it did. */
export function useModel(kind: "image" | "edit", allow = true, builtIn = false): [string, ReactNode] {
  const [all, setAll] = useState<Workflow[]>([]);
  useEffect(() => {
    (asked ??= api.workflows().catch(() => [])).then(setAll);
  }, []);
  const [saved, save] = useStored(`effigy.model.${kind}`, "");
  const list = all.filter((w) => w.kind === kind && (builtIn || !w.reference));
  const usable = list.filter((w) => !w.missing.length);
  const id = !allow ? "" : usable.some((w) => w.id === saved) ? saved : kind === "edit" && !builtIn ? (usable[0]?.id ?? "") : "";
  const w = list.find((x) => x.id === id);
  const lacking = list.filter((x) => x.missing.length);
  if (!list.length) return ["", null];
  const picker = (
    <>
      <label>
        {kind === "edit" ? "Edit model" : "Image model"}
        <select value={id} disabled={!allow} onChange={(e) => save(e.target.value)}>
          {(kind === "image" || builtIn) && <option value="">{builtIn ? "Z-Image + Tile (built in)" : "Z-Image Turbo (built in)"}</option>}
          {list.map((x) => (
            <option key={x.id} value={x.id} disabled={!!x.missing.length}>
              {x.title}
              {x.own ? " (yours)" : ""}
              {x.missing.length ? " (files missing)" : ""}
            </option>
          ))}
        </select>
      </label>
      {w?.about && <p className="muted">{w.about}</p>}
      {w?.restricted && (
        <p className="note warn">
          Not for commercial use:{" "}
          {w.models
            .filter((m) => m.commercial !== true)
            .map((m) => `${m.name} (${m.license || "licence not stated"})`)
            .join(", ") || "this workflow does not list its models"}
          . What you make with it is marked restricted: fine for prototypes and personal projects.
        </p>
      )}
      {lacking.length > 0 && (
        <details className="muted">
          <summary>Models with missing files</summary>
          <ul>
            {lacking.map((x) => (
              <li key={x.id}>
                {x.title}: {x.missing.join(", ")}
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
  return [id, picker];
}

/** The words Effigy adds to a description, open to change: [brief's extra params, concepts' extra params, the panel].
 * Untouched, both are empty and the steps do what they always did. A change is kept per kind in this browser. */
export function usePromptParts(category: string, prompt: string): [{ framing?: string }, { variants?: false }, ReactNode] {
  const [templates, setTemplates] = useState<Record<string, string>>({});
  useEffect(() => {
    (framings ??= api.framings().catch(() => ({}))).then(setTemplates);
  }, []);
  const [own, setOwn] = useStored<string | null>(`effigy.framing.${category}`, null);
  const [vary, setVary] = useStored("effigy.variants", true);
  const framing = own ?? templates[category] ?? "";
  const field = useId(); // the label beside the field, not around it: its name is not the words in it
  const panel = (
    <details>
      <summary>Prompt details{own !== null || !vary ? " (changed)" : ""}</summary>
      <div className="stack">
        <label htmlFor={field}>Added to your prompt</label>
        <textarea id={field} rows={4} value={framing} onChange={(e) => setOwn(e.target.value)} />
        {own !== null && (
          <button className="small" onClick={() => setOwn(null)}>
            Back to this kind's own words
          </button>
        )}
        <label className="check">
          <input type="checkbox" checked={vary} onChange={(e) => setVary(e.target.checked)} /> Vary the four pictures (a wider frame, a studio shot, crisper
          detail)
        </label>
        <p className="muted">The image model gets: {[prompt.trim() || "your description", framing.trim()].filter(Boolean).join(", ")}</p>
      </div>
    </details>
  );
  return [own === null ? {} : { framing: own }, vary ? {} : { variants: false }, panel];
}

/** Edit picture: the shown picture (or one of a version's several) redrawn with a change put in words. */
export function EditPicture(props: { asset: string; version: Meta; chain: Chain; picked?: string }) {
  const v = props.version;
  const keys = Object.keys(v.outputs)
    .filter((k) => /^(cand|image)_\d+$/.test(k))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const [chosen, setChosen] = useState("");
  const key = !keys.length ? undefined : keys.includes(props.picked ?? "") ? props.picked : keys.includes(chosen) ? chosen : keys[0];
  const [prompt, setPrompt] = useState("");
  const [model, picker] = useModel("edit");
  const [busy, run] = useBusy();
  if (!picker)
    return (
      <p className="muted">
        Changing a picture with words runs through ComfyUI and an edit model: turn ComfyUI on in <a href="#/setup">Setup</a>.
      </p>
    );
  return (
    <div className="stack">
      {keys.length > 1 && (
        <div className="row thumbs" role="group" aria-label="Picture to edit">
          {keys.map((k) => (
            <button key={k} aria-pressed={k === key} onClick={() => setChosen(k)}>
              <img src={fileUrl(props.asset, v.version, v.outputs[k])} alt={`picture ${Number(k.split("_")[1]) + 1}`} />
            </button>
          ))}
        </div>
      )}
      <label>
        What to change
        <textarea
          rows={3}
          placeholder="Give him a horned iron helmet. Keep the pose, the face and the plain background."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </label>
      {picker}
      <button
        className="primary"
        disabled={busy || !prompt.trim() || !model}
        onClick={() => run(() => props.chain(props.asset, v.version, [["edit", { prompt: prompt.trim(), workflow: model, ...(key ? { key } : {}) }]]))}
      >
        {busy ? "Starting…" : "Edit the picture"}
      </button>
      <p className="muted">The result is a new picture; this one stays. The model redraws all of it, so say what must stay the same.</p>
    </div>
  );
}
