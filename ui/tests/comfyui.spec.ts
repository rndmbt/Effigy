import { test, expect } from "@playwright/test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

test("ComfyUI folder choice persists without moving native models", async ({ page, request }) => {
  const base = mkdtempSync(join(tmpdir(), "effigy-comfy-folders-"));
  const shared = join(base, "models");
  mkdirSync(shared);
  const keep = join(shared, "keep.txt");
  writeFileSync(keep, "existing shared model store");
  const main = await (await request.get("/api/setup/folder")).json();
  try {
    await page.goto("/#/setup");
    const panel = page.getByRole("region", { name: "ComfyUI backend", exact: true });
    await panel.getByLabel("Set up ComfyUI automatically").check();
    await panel.getByLabel("ComfyUI models folder", { exact: true }).fill(shared);
    await panel.getByRole("button", { name: "Save ComfyUI settings" }).click();
    await expect.poll(async () => (await (await request.get("/api/setup/comfyui")).json()).models).toBe(shared);
    await page.reload();
    await expect(panel.getByLabel("ComfyUI models folder", { exact: true })).toHaveValue(shared);
    expect((await (await request.get("/api/setup/comfyui")).json()).download_path).toBe(shared);
    expect((await (await request.get("/api/setup/folder")).json()).path).toBe(main.path);
    expect(readFileSync(keep, "utf8")).toBe("existing shared model store");
    await panel.getByLabel("ComfyUI models folder", { exact: true }).fill("");
    await panel.getByRole("button", { name: "Save ComfyUI settings" }).click();
    await expect.poll(async () => (await (await request.get("/api/setup/comfyui")).json()).download_path).toBe(main.path);
  } finally {
    await request.post("/api/setup/comfyui", { data: { enabled: false, models: "" } });
    rmSync(base, { recursive: true });
  }
});

test("Guided setup saves ComfyUI folder before installing it", async ({ page }) => {
  const secondary = join(tmpdir(), "shared-comfy-models");
  let state = { enabled: false, models: "", installed: false, installing: false, download_path: "" };
  const calls: string[] = [];
  await page.route("**/api/setup/comfyui", async (route) => {
    if (route.request().method() === "POST") {
      calls.push("save"); state = { ...state, ...route.request().postDataJSON() };
    }
    await route.fulfill({ json: state });
  });
  await page.route("**/api/setup/comfyui/install", (r) => {
    calls.push("install");
    return r.fulfill({ status: 202, json: { job_id: "comfy-install" } });
  });
  await page.route("**/api/jobs/comfy-install", (r) => r.fulfill({ json: { status: "done" } }));
  await page.route("**/api/setup/presets", (r) => r.fulfill({ json: [{ id: "characters", name: "Game characters", groups: [], bytes: 0, missing: 0 }] }));
  await page.route("**/api/environments", (r) => r.fulfill({ json: [] }));
  await page.route("**/api/hf", (r) => r.fulfill({ json: { token: false, user: null, access: {}, gated: {} } }));
  await page.route("**/api/setup/welcomed", (r) => r.fulfill({ json: { ok: true } }));
  await page.goto("/#/welcome");
  const panel = page.getByRole("region", { name: "ComfyUI backend", exact: true });
  await expect(panel.getByLabel("Set up ComfyUI automatically")).not.toBeChecked();
  await panel.getByLabel("Set up ComfyUI automatically").check();
  await panel.getByLabel("ComfyUI models folder", { exact: true }).fill(secondary);
  await page.getByRole("button", { name: /^Install Game characters/ }).click();
  await expect.poll(() => calls).toEqual(["save", "install"]);
  expect(state.models).toBe(secondary);
});
