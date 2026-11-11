import { test, expect } from "@playwright/test";

const flows = [
  { id: "fast", title: "Fast model", kind: "image", about: "Eight steps.", restricted: true, own: false, missing: [], models: [{ name: "lab/fast", license: "research only", commercial: false }] },
  { id: "big", title: "Big model", kind: "image", about: "", restricted: false, own: true, missing: ["big.safetensors"], models: [] },
];

test("a ComfyUI workflow and changed prompt parts ride on the brief and the concepts", async ({ page }) => {
  await page.route("**/api/workflows", (r) => r.fulfill({ json: flows }));
  await page.route("**/api/framings", (r) => r.fulfill({ json: { prop: "centred, plain background", character: "full body, A-pose" } }));
  const sent: Record<string, any> = {};
  await page.route("**/api/assets/*/steps/concepts", async (route) => {
    sent.concepts = route.request().postDataJSON().params;
    await route.fulfill({ status: 202, json: { job_id: "none", asset_id: "x", status: "queued" } });
  });
  await page.route("**/api/jobs/none", (r) => r.fulfill({ json: { job_id: "none", status: "failed", error: { code: "failed", message: "not run in this test" } } }));
  page.on("request", (req) => {
    if (req.method() === "POST" && req.url().endsWith("/steps/brief")) sent.brief = req.postDataJSON().params;
  });
  await page.goto("/#/new");
  await page.getByLabel("Description").fill("a brass lantern");
  const model = page.getByLabel("Image model");
  await expect(model.locator("option")).toHaveText(["Z-Image Turbo (built in)", "Fast model", "Big model (yours) (files missing)"]);
  await expect(model.locator("option").nth(2)).toHaveAttribute("disabled", "");
  await model.selectOption("fast");
  await expect(page.getByText("Not for commercial use: lab/fast (research only)")).toBeVisible();
  await page.getByText("Prompt details").click();
  const framing = page.getByLabel("Added to your prompt");
  await expect(framing).toHaveValue("centred, plain background");
  await framing.fill("side view, on white");
  await expect(page.getByText("The image model gets: a brass lantern, side view, on white")).toBeVisible();
  await page.getByLabel(/Vary the four pictures/).uncheck();
  await page.getByRole("button", { name: "Generate 4 concepts" }).click();
  await expect.poll(() => sent.concepts).toEqual({ variants: false, workflow: "fast" });
  expect(sent.brief).toEqual({ prompt: "a brass lantern", category: "prop", framing: "side view, on white" });
});

test("without ComfyUI the form has no model list and sends what it always did", async ({ page }) => {
  let params: any;
  await page.route("**/api/assets/*/steps/concepts", async (route) => {
    params = route.request().postDataJSON().params;
    await route.fulfill({ status: 202, json: { job_id: "none", asset_id: "x", status: "queued" } });
  });
  await page.route("**/api/jobs/none", (r) => r.fulfill({ json: { job_id: "none", status: "failed", error: { code: "failed", message: "not run in this test" } } }));
  await page.route("**/api/models", (r) => r.fulfill({ json: [] })); // nothing known missing: the Generate button shows
  await page.goto("/#/new");
  await page.getByLabel("Description").fill("a brass lantern");
  await expect(page.getByText("Prompt details")).toBeVisible();
  await expect(page.getByLabel("Image model")).toHaveCount(0);
  await page.getByRole("button", { name: "Generate 4 concepts" }).click();
  await expect.poll(() => params).toEqual({});
});
