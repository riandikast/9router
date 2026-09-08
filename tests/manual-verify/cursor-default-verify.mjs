// Minimal vitest-free harness replicating tests/unit/cursor-default.test.js
// and tests/unit/cursor-composer-thinking.test.js assertions 1:1.
import assert from "node:assert";

const { CursorExecutor } = await import("../../open-sse/executors/cursor.js");
const { encodeField, wrapConnectRPCFrame } = await import("../../open-sse/utils/cursorProtobuf.js");
const {
  normalizeCursorModelId,
  resolveCursorUpstreamModel,
  shouldPromoteThinkingToContent,
  visibleContentFromThinking,
} = await import("../../open-sse/utils/cursorModel.js");

const LEN = 2;

function cursorResponseFrame({ text = "", thinking = "" }) {
  const responseFields = [];

  if (text) {
    responseFields.push(encodeField(1, LEN, text));
  }

  if (thinking) {
    const thinkingMessage = encodeField(1, LEN, thinking);
    responseFields.push(encodeField(25, LEN, thinkingMessage));
  }

  const response = Buffer.concat(responseFields.map((field) => Buffer.from(field)));
  const envelope = encodeField(2, LEN, response);
  return Buffer.from(wrapConnectRPCFrame(envelope));
}

function parseSSE(text) {
  return text
    .split("\n\n")
    .filter((chunk) => chunk.startsWith("data: "))
    .map((chunk) => chunk.slice("data: ".length))
    .filter((data) => data !== "[DONE]")
    .map((data) => JSON.parse(data));
}

const results = [];
async function test(name, fn) {
  try {
    await fn();
    results.push(`  PASS  ${name}`);
  } catch (err) {
    results.push(`  FAIL  ${name}\n        ${err.message}`);
    process.exitCode = 1;
  }
}

// ---- cursorModel helpers (PR test file 1) ----
await test("normalizes legacy Claude model ids", () => {
  assert.equal(normalizeCursorModelId("cu/claude-3-5-sonnet-20240620"), "claude-4.5-sonnet");
});

await test("resolves default/auto to upstream fallback", () => {
  assert.equal(resolveCursorUpstreamModel("cu/default"), "claude-4.5-sonnet");
  assert.equal(resolveCursorUpstreamModel("auto"), "claude-4.5-sonnet");
});

await test("promotes thinking for default and -thinking models", () => {
  assert.equal(shouldPromoteThinkingToContent("cu/default"), true);
  assert.equal(shouldPromoteThinkingToContent("claude-4.5-sonnet-thinking"), true);
  assert.equal(shouldPromoteThinkingToContent("gpt-5.3-codex"), false);
});

await test("visibleContentFromThinking extracts text after </think>", () => {
  assert.equal(visibleContentFromThinking("a</think>b"), "b");
  assert.equal(visibleContentFromThinking("no end tag"), "");
});

// ---- CursorExecutor default model responses (PR test file 2) ----
await test("uses visible content after </think> for default model", async () => {
  const executor = new CursorExecutor();
  const buffer = cursorResponseFrame({ thinking: "internal plan</think>Hello!" });

  const response = executor.transformProtobufToJSON(buffer, "cu/default", {
    messages: [{ role: "user", content: "hi" }],
  });
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.choices[0].message.content, "Hello!");
  assert.ok(payload.usage.completion_tokens > 0);
});

await test("uses visible thinking for -thinking variant models", async () => {
  const executor = new CursorExecutor();
  const buffer = cursorResponseFrame({ thinking: "reasoning</think>Visible" });

  const response = executor.transformProtobufToJSON(buffer, "claude-4.5-sonnet-thinking", {
    messages: [{ role: "user", content: "hi" }],
  });
  const payload = await response.json();

  assert.equal(payload.choices[0].message.content, "Visible");
});

await test("returns error for empty completion with no tool calls", async () => {
  const executor = new CursorExecutor();
  const response = executor.transformProtobufToJSON(Buffer.alloc(0), "cu/default", {
    messages: [{ role: "user", content: "hi" }],
  });
  const payload = await response.json();

  assert.equal(response.status, 502);
  assert.equal(payload.error?.code, "empty_completion");
});

await test("returns error for thinking-only on non-promoted models", async () => {
  const executor = new CursorExecutor();
  const buffer = cursorResponseFrame({ thinking: "private</think>hidden" });

  const response = executor.transformProtobufToJSON(buffer, "gpt-5.3-codex", {
    messages: [{ role: "user", content: "hi" }],
  });
  const payload = await response.json();

  assert.equal(response.status, 502);
  assert.equal(payload.error?.code, "empty_completion");
  assert.ok(!JSON.stringify(payload).includes("hidden"));
});

// ---- Composer regression (existing test file) ----
await test("Composer non-streaming: visible content only, no reasoning leak", async () => {
  const executor = new CursorExecutor();
  const buffer = cursorResponseFrame({
    thinking: "private reasoning that must not leak</think>OK",
  });

  const response = executor.transformProtobufToJSON(buffer, "cu/composer-2.5", {
    messages: [{ role: "user", content: "reply OK" }],
  });
  const payload = await response.json();

  assert.equal(payload.choices[0].message.content, "OK");
  assert.ok(!JSON.stringify(payload).includes("private reasoning"));
  assert.ok(payload.usage.completion_tokens > 0);
});

await test("Composer streaming: only visible content streamed", async () => {
  const executor = new CursorExecutor();
  const buffer = Buffer.concat([
    cursorResponseFrame({ thinking: "private reasoning" }),
    cursorResponseFrame({ thinking: " that must not leak</think>O" }),
    cursorResponseFrame({ thinking: "K" }),
  ]);

  const response = executor.transformProtobufToSSE(buffer, "composer-2.5-fast", {
    messages: [{ role: "user", content: "reply OK" }],
  });
  const events = parseSSE(await response.text());
  const content = events.map((e) => e.choices?.[0]?.delta?.content || "").join("");

  assert.equal(content, "OK");
  assert.ok(!JSON.stringify(events).includes("private reasoning"));
  assert.ok(events.at(-1).usage.completion_tokens > 0);
});

// ---- streaming default-model (SSE) empty guard ----
await test("SSE default model: thinking-only promotes visible text", async () => {
  const executor = new CursorExecutor();
  const buffer = cursorResponseFrame({ thinking: "plan</think>Hi there" });

  const response = executor.transformProtobufToSSE(buffer, "cu/default", {
    messages: [{ role: "user", content: "hi" }],
  });
  const events = parseSSE(await response.text());
  const content = events.map((e) => e.choices?.[0]?.delta?.content || "").join("");

  assert.equal(content, "Hi there");
});

await test("SSE empty completion returns 502 empty_completion", async () => {
  const executor = new CursorExecutor();
  const response = executor.transformProtobufToSSE(Buffer.alloc(0), "cu/default", {
    messages: [{ role: "user", content: "hi" }],
  });
  const payload = await response.json();

  assert.equal(response.status, 502);
  assert.equal(payload.error?.code, "empty_completion");
});

// ---- AgentService port: default resolution in executeAgent ----
await test("executeAgent resolves default → claude-4.5-sonnet upstream (buildAgentRunFrame spy)", async () => {
  const sentModels = [];
  const executor = new CursorExecutor();

  // Spy on the h2 session: capture the frame written for the run request.
  const fakeSession = {
    write: (frame) => sentModels.push(frame),
    end: () => {},
    close: () => {},
    responseHeaders: Promise.resolve({ ":status": 200 }),
    read: async () => ({ value: undefined, done: true }),
  };
  executor.openAgentHttp2Stream = () => fakeSession;
  executor.buildHeaders = () => ({});

  const result = await executor.executeAgent({
    model: "cu/default",
    body: { messages: [{ role: "user", content: "hi" }] },
    stream: false,
    credentials: { accessToken: "x", providerSpecificData: { machineId: "m" } },
    signal: null,
  });

  // Decode the single frame written: extract the requested model string.
  const { decodeMessage } = await import("../../open-sse/utils/cursorProtobuf.js");
  const payload = sentModels[0].subarray(5); // strip connect frame header
  const serverMessage = decodeMessage(payload);
  const runRequest = decodeMessage(serverMessage.get(1)[0].value);
  const requestedModel = decodeMessage(runRequest.get(9)[0].value);
  const modelStr = Buffer.from(requestedModel.get(1)[0].value).toString("utf8");

  assert.equal(modelStr, "claude-4.5-sonnet");
  assert.equal(result.response.status, 502); // empty turn (fake session) → explicit error
  const body = await result.response.json();
  assert.equal(body.error.code, "empty_completion");
});

await test("executeAgent passes concrete model through unchanged", async () => {
  const sentModels = [];
  const executor = new CursorExecutor();

  const fakeSession = {
    write: (frame) => sentModels.push(frame),
    end: () => {},
    close: () => {},
    responseHeaders: Promise.resolve({ ":status": 200 }),
    read: async () => ({ value: undefined, done: true }),
  };
  executor.openAgentHttp2Stream = () => fakeSession;
  executor.buildHeaders = () => ({});

  await executor.executeAgent({
    model: "cu/kimi-k2.5",
    body: { messages: [{ role: "user", content: "hi" }] },
    stream: false,
    credentials: { accessToken: "x", providerSpecificData: { machineId: "m" } },
    signal: null,
  });

  const { decodeMessage } = await import("../../open-sse/utils/cursorProtobuf.js");
  const payload = sentModels[0].subarray(5);
  const serverMessage = decodeMessage(payload);
  const runRequest = decodeMessage(serverMessage.get(1)[0].value);
  const requestedModel = decodeMessage(runRequest.get(9)[0].value);
  const modelStr = Buffer.from(requestedModel.get(1)[0].value).toString("utf8");

  assert.equal(modelStr, "kimi-k2.5");
});

console.log(results.join("\n"));
console.log(`\n${results.filter((r) => r.includes("PASS")).length}/${results.length} passed`);
