// OpenAI-compatible RPC shim: implements the same interface as CodexRpc
// (request/notify/respond/respondError/start/stop + 'notification'/'request'/'exit' events)
// but backs the ~13 RPC methods the Brain/Bridge actually use with an
// OpenAI-compatible chat completions API (e.g. Lingsuan) instead of `codex app-server`.
//
// Env:
//   OPA_OPENAI_BASE_URL  default https://lingsuan.top/v1
//   OPA_OPENAI_API_KEY   required
//   OPA_OPENAI_MODEL     default gpt-6-sol
import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';
import { log } from '../util/log.ts';

interface OAITool { type: 'function'; function: { name: string; description?: string; parameters?: unknown } }
interface Session {
  id: string;
  model: string;
  instructions: string;
  tools: OAITool[];
  messages: any[];
  turn?: { id: string; controller: AbortController; finish: (status: string) => void };
  steerQueue: string[];
}

const MAX_TOOL_ROUNDS = 24;

export class OpenAIRpc extends EventEmitter {
  private sessions = new Map<string, Session>();
  private nextReqId = 1;
  private pendingRespond = new Map<number | string, (v: any) => void>();
  private baseUrl: string;
  private apiKey: string;
  private defaultModel: string;

  constructor() {
    super();
    this.baseUrl = (process.env.OPA_OPENAI_BASE_URL ?? 'https://lingsuan.top/v1').replace(/\/$/, '');
    this.apiKey = process.env.OPA_OPENAI_API_KEY ?? '';
    this.defaultModel = process.env.OPA_OPENAI_MODEL ?? 'gpt-6-sol';
    if (!this.apiKey) log.warn('openai', 'OPA_OPENAI_API_KEY not set; model calls will fail');
  }

  start(): void { log.info('openai', `OpenAI RPC ready (${this.baseUrl}, model ${this.defaultModel})`); }
  stop(): void {
    for (const s of this.sessions.values()) s.turn?.controller.abort();
  }

  async request<T = any>(method: string, params?: any, timeoutMs = 120_000): Promise<T> {
    switch (method) {
      case 'initialize': return { capabilities: { openaiShim: true } } as T;
      case 'thread/start': return this.threadStart(params) as T;
      case 'thread/resume': return this.threadResume(params) as T;
      case 'thread/archive': this.sessions.delete(params?.threadId); return {} as T;
      case 'thread/name/set': return {} as T;
      case 'thread/unsubscribe': return {} as T;
      case 'thread/goal/set': return {} as T;
      case 'thread/compact/start': return this.threadCompact(params, timeoutMs) as T;
      case 'turn/start': return this.turnStart(params, timeoutMs) as T;
      case 'turn/steer': return this.turnSteer(params) as T;
      case 'turn/interrupt': return this.turnInterrupt(params) as T;
      case 'mcpServer/tool/call': throw new Error('mcpServer/tool/call not supported in OpenAI mode');
      default: throw new Error(`OpenAIRpc: unsupported method ${method}`);
    }
  }

  notify(): void { /* no-op */ }

  respond(id: number | string, result: unknown) {
    const r = this.pendingRespond.get(id);
    if (r) { this.pendingRespond.delete(id); r(result); }
  }
  respondError(id: number | string, _code: number, message: string) {
    this.respond(id, { success: false, contentItems: [{ type: 'inputText', text: message }] });
  }

  // ── threads ──

  private threadStart(p: any) {
    const id = randomUUID();
    const tools: OAITool[] = (p?.dynamicTools ?? []).map((t: any) => ({
      type: 'function',
      function: { name: t.name, description: t.description ?? '', parameters: t.inputSchema ?? { type: 'object', properties: {} } },
    }));
    const model = p?.model || this.defaultModel;
    const instructions = p?.baseInstructions ?? p?.developerInstructions ?? '';
    const s: Session = {
      id, model, instructions, tools, steerQueue: [],
      messages: instructions ? [{ role: 'system', content: instructions }] : [],
    };
    this.sessions.set(id, s);
    log.info('openai', `thread started ${id} (model ${model}, ${tools.length} tools)`);
    return { thread: { id }, model };
  }

  private threadResume(p: any) {
    const s = this.sessions.get(p?.threadId);
    if (!s) throw new Error(`unknown thread ${p?.threadId}`);
    return { thread: { id: s.id }, model: s.model };
  }

  private async threadCompact(p: any, timeoutMs: number) {
    const s = this.sessions.get(p?.threadId);
    if (!s) throw new Error(`unknown thread ${p?.threadId}`);
    // Ask the model to summarize, then replace history with the summary.
    const summary = await this.chat(s, [
      ...s.messages,
      { role: 'user', content: '请把以上对话压缩成一份简明摘要（保留关键事实、待办和用户偏好），只输出摘要正文。' },
    ], [], timeoutMs);
    s.messages = [
      ...(s.instructions ? [{ role: 'system', content: s.instructions }] : []),
      { role: 'user', content: `<context_summary>\n${summary}\n</context_summary>` },
    ];
    this.emit('notification', 'thread/compacted', { threadId: s.id });
    return { ok: true };
  }

  // ── turns ──

  private async turnStart(p: any, timeoutMs: number) {
    const s = this.sessions.get(p?.threadId);
    if (!s) throw new Error(`unknown thread ${p?.threadId}`);
    if (s.turn) throw new Error('turn already active');
    const turnId = randomUUID();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let finish!: (status: string) => void;
    const done = new Promise<string>((r) => { finish = r; });
    s.turn = { id: turnId, controller, finish };
    const texts = (p?.input ?? []).filter((i: any) => i?.type === 'text').map((i: any) => String(i.text ?? ''));
    const model = p?.model || s.model;
    this.emit('notification', 'turn/started', { threadId: s.id, turn: { id: turnId } });
    void this.runTurn(s, turnId, texts.join('\n'), model, controller).then(
      (status) => { clearTimeout(timer); this.endTurn(s, turnId, status); finish(status); },
      (e) => { clearTimeout(timer); this.endTurn(s, turnId, 'failed', String(e?.message ?? e)); finish('failed'); },
    );
    return { turn: { id: turnId } };
  }

  private endTurn(s: Session, turnId: string, status: string, error?: string) {
    if (s.turn?.id !== turnId) return;
    s.turn = undefined;
    // drain steered inputs as a follow-up user message is handled by the caller via send()
    this.emit('notification', 'turn/completed', { threadId: s.id, turn: { id: turnId, status, error: error ? { message: error } : undefined } });
  }

  private async turnSteer(p: any) {
    const s = this.sessions.get(p?.threadId);
    if (!s?.turn) return false;
    const texts = (p?.input ?? []).filter((i: any) => i?.type === 'text').map((i: any) => String(i.text ?? ''));
    if (texts.length) s.steerQueue.push(texts.join('\n'));
    return true;
  }

  private async turnInterrupt(p: any) {
    const s = this.sessions.get(p?.threadId);
    if (!s?.turn) return false;
    s.turn.controller.abort();
    return true;
  }

  // ── core agentic loop ──

  private async runTurn(s: Session, turnId: string, userText: string, model: string, controller: AbortController): Promise<string> {
    if (userText) s.messages.push({ role: 'user', content: userText });
    // drain any steered input that arrived before the loop started
    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      if (controller.signal.aborted) return 'interrupted';
      if (s.steerQueue.length) {
        s.messages.push({ role: 'user', content: s.steerQueue.splice(0).join('\n') });
      }
      const { text, toolCalls } = await this.chatStream(s, model, controller, turnId);
      if (controller.signal.aborted) return 'interrupted';
      if (!toolCalls.length) {
        s.messages.push({ role: 'assistant', content: text });
        this.emit('notification', 'item/completed', {
          threadId: s.id, item: { type: 'agentMessage', text, id: randomUUID() },
        });
        return 'completed';
      }
      s.messages.push({ role: 'assistant', content: text || null, tool_calls: toolCalls });
      for (const tc of toolCalls) {
        if (controller.signal.aborted) return 'interrupted';
        this.emit('notification', 'item/completed', {
          threadId: s.id,
          item: { type: 'toolCall', name: tc.function.name, id: tc.id },
        });
        const resultText = await this.invokeTool(s.id, tc.function.name, tc.function.arguments);
        s.messages.push({ role: 'tool', tool_call_id: tc.id, content: resultText });
      }
    }
    return 'completed';
  }

  /** Route a model tool call through the Bridge's dynamicToolHandler via the 'request' event. */
  private invokeTool(threadId: string, tool: string, argsJson: string): Promise<string> {
    const id = this.nextReqId++;
    let args: any = {};
    try { args = JSON.parse(argsJson || '{}'); } catch { /* keep {} */ }
    return new Promise<string>((resolve) => {
      const timer = setTimeout(() => { this.pendingRespond.delete(id); resolve('工具调用超时'); }, 110_000);
      this.pendingRespond.set(id, (r: any) => {
        clearTimeout(timer);
        const items = r?.contentItems ?? [];
        const text = items.map((c: any) => (typeof c?.text === 'string' ? c.text : JSON.stringify(c))).join('\n');
        resolve(r?.success === false ? `工具失败: ${text}` : (text || '(无输出)'));
      });
      this.emit('request', id, 'item/tool/call', { threadId, tool, arguments: args, callId: String(id) });
    });
  }

  private async chat(s: Session, messages: any[], tools: OAITool[], timeoutMs: number): Promise<string> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await this.postChat(messages, tools, false, controller.signal);
      const j: any = await res.json();
      return j.choices?.[0]?.message?.content ?? '';
    } finally { clearTimeout(timer); }
  }

  private async chatStream(s: Session, model: string, controller: AbortController, turnId: string): Promise<{ text: string; toolCalls: any[] }> {
    const res = await this.postChat(s.messages, s.tools, true, controller.signal, model);
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let text = '';
    const toolCalls: any[] = [];
    const itemId = randomUUID();
    const pushDelta = (d: string) => {
      if (!d) return;
      text += d;
      this.emit('notification', 'item/agentMessage/delta', { threadId: s.id, itemId, delta: d, turnId });
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') break;
        let ev: any;
        try { ev = JSON.parse(data); } catch { continue; }
        const delta = ev.choices?.[0]?.delta;
        if (!delta) continue;
        pushDelta(delta.content ?? '');
        for (const tc of delta.tool_calls ?? []) {
          const cur = toolCalls[tc.index ?? 0] ??= { id: '', type: 'function', function: { name: '', arguments: '' } };
          if (tc.id) cur.id = tc.id;
          if (tc.function?.name) cur.function.name += tc.function.name;
          if (tc.function?.arguments) cur.function.arguments += tc.function.arguments;
        }
      }
      if (controller.signal.aborted) break;
    }
    return { text, toolCalls: toolCalls.filter((t) => t.function?.name) };
  }

  private async postChat(messages: any[], tools: OAITool[], stream: boolean, signal: AbortSignal, model?: string) {
    const body: any = {
      model: model ?? this.defaultModel,
      messages,
      stream,
      ...(tools.length ? { tools, tool_choice: 'auto' } : {}),
    };
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const t = await res.text().catch(() => '');
      throw new Error(`OpenAI API ${res.status}: ${t.slice(0, 300)}`);
    }
    return res;
  }
}
