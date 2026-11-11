import { useEffect, useState } from "react";
import { api, type ComfyUIState, type JobStatus } from "./api";

export function ComfyUISetup(props: {
  jobs: Record<string, { step?: string; status: JobStatus; text?: string; error?: { message: string } }>;
  track: (id: string, step: string) => void;
  guard: (p: Promise<unknown>) => Promise<void>;
  guided?: boolean;
  onChange?: (config: ComfyUIState | null) => void;
}) {
  const [config, setConfig] = useState<ComfyUIState | null>(null);
  const [pending, setPending] = useState(false);
  const job = Object.values(props.jobs).filter((j) => j.step === "install comfyui").at(-1);
  const busy = pending || config?.installing || job?.status === "running" || job?.status === "queued";
  useEffect(() => { props.guard(api.comfyUI().then(setConfig)); }, [job?.status]);
  useEffect(() => { props.onChange?.(config); }, [config]);
  if (!config) return <p className="muted">Checking ComfyUI installation…</p>;
  const save = () => api.setComfyUI({ enabled: config.enabled, models: config.models }).then(setConfig);
  const act = (install: boolean) => {
    setPending(true);
    props.guard((async () => {
      await save();
      if (install) {
        const { job_id } = await api.installComfyUI();
        props.track(job_id, "install comfyui");
      }
    })().finally(() => setPending(false)));
  };
  return <section aria-label="ComfyUI backend" className="stack">
    <h2>ComfyUI backend</h2>
    <label className="check"><input type="checkbox" checked={config.enabled} disabled={!!busy}
      onChange={(e) => setConfig({ ...config, enabled: e.target.checked })} />Set up ComfyUI automatically</label>
    {config.enabled && <>
      <label>ComfyUI models folder (optional)
        <div className="row"><input aria-label="ComfyUI models folder" placeholder="Blank: use Effigy's models folder" value={config.models} disabled={!!busy}
          onChange={(e) => setConfig({ ...config, models: e.target.value })} />
          <button disabled={!!busy} onClick={() => props.guard(api.pick("folder").then(({ path }) => { if (path) setConfig({ ...config, models: path }); }))}>Browse ComfyUI models</button>
        </div>
      </label>
      <p className="muted">Leave blank to share Effigy's models folder. With a second folder, ComfyUI searches both but downloads new models only into the second. Effigy's own downloads stay in its main folder.</p>
    </>}
    <p className="status">{busy ? "Installing ComfyUI…" : job?.status === "failed" ? `Install failed: ${job.error?.message ?? "see the activity log"}` : config.installed ? "ComfyUI is installed" : "ComfyUI is not installed"}</p>
    {props.guided ? <p className="muted">Included with the Install button below; no model weights are bundled.</p> : <div className="row">
      <button disabled={!!busy} onClick={() => act(false)}>Save ComfyUI settings</button>
      {config.enabled && !config.installed && <button disabled={!!busy} onClick={() => act(true)}>Install ComfyUI</button>}
    </div>}
  </section>;
}
