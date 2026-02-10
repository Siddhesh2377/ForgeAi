import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface TestResult {
  text: string;
  tokens_generated: number;
  time_ms: number;
  device: string;
}

export interface GenerateOptions {
  modelPath: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP?: number | null;
  topK?: number | null;
  repeatPenalty?: number | null;
  gpuLayers?: number | null;
  systemPrompt?: string | null;
  contextSize?: number | null;
}

class TestStore {
  generating = $state(false);
  output = $state("");
  error = $state<string | null>(null);
  result = $state<TestResult | null>(null);

  private tokenUnlisten: UnlistenFn | null = null;

  async generate(opts: GenerateOptions) {
    if (!this.tokenUnlisten) {
      this.tokenUnlisten = await listen<string>("test:token", (e) => {
        this.output += e.payload;
      });
    }

    this.generating = true;
    this.output = "";
    this.error = null;
    this.result = null;

    try {
      this.result = await invoke<TestResult>("test_generate", {
        modelPath: opts.modelPath,
        prompt: opts.prompt,
        maxTokens: opts.maxTokens,
        temperature: opts.temperature,
        topP: opts.topP ?? null,
        topK: opts.topK ?? null,
        repeatPenalty: opts.repeatPenalty ?? null,
        gpuLayers: opts.gpuLayers ?? null,
        systemPrompt: opts.systemPrompt ?? null,
        contextSize: opts.contextSize ?? null,
      });
    } catch (e) {
      const msg = String(e);
      if (!msg.includes("cancelled")) {
        this.error = msg;
      }
    } finally {
      this.generating = false;
    }
  }

  async cancel() {
    try {
      await invoke("test_cancel");
    } catch {
      // ignore
    }
  }

  clear() {
    this.output = "";
    this.error = null;
    this.result = null;
  }
}

export const test = new TestStore();
