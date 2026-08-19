import type { Message, MessagePart } from "./types";

export type MessageBlock =
  | { kind: "markdown"; key: string; text: string }
  | { kind: "part"; key: string; part: MessagePart };

function finiteNumber(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function timestamp(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/** Keep ChatStorage sequence authoritative and use timestamps only for legacy payloads. */
export function normalizeMessages(messages: Message[]): Message[] {
  return messages
    .map((message, originalIndex) => ({
      message: {
        ...message,
        parts: [...(message.parts || [])]
          .map((part, partIndex) => ({ part, partIndex }))
          .sort((left, right) => {
            const leftSeq = finiteNumber(left.part.part_seq);
            const rightSeq = finiteNumber(right.part.part_seq);
            if (
              leftSeq !== undefined &&
              rightSeq !== undefined &&
              leftSeq !== rightSeq
            )
              return leftSeq - rightSeq;
            return left.partIndex - right.partIndex;
          })
          .map(({ part }) => part),
      },
      originalIndex,
    }))
    .sort((left, right) => {
      const leftSeq = finiteNumber(left.message.seq);
      const rightSeq = finiteNumber(right.message.seq);
      if (
        leftSeq !== undefined &&
        rightSeq !== undefined &&
        leftSeq !== rightSeq
      )
        return leftSeq - rightSeq;

      const leftTime = timestamp(left.message.created_at);
      const rightTime = timestamp(right.message.created_at);
      if (
        leftTime !== undefined &&
        rightTime !== undefined &&
        leftTime !== rightTime
      )
        return leftTime - rightTime;
      return left.originalIndex - right.originalIndex;
    })
    .map(({ message }) => message);
}

/**
 * Merge only adjacent text parts into one Markdown document. Tables, tool calls and statuses
 * remain in their exact part_seq positions, so the stored execution narrative is not changed.
 */
export function buildMessageBlocks(parts: MessagePart[]): MessageBlock[] {
  const blocks: MessageBlock[] = [];
  for (const part of parts) {
    if (part.kind !== "text") {
      blocks.push({ kind: "part", key: `part-${part.part_seq}`, part });
      continue;
    }

    const text = String(part.payload?.text || "");
    const previous = blocks.at(-1);
    if (previous?.kind === "markdown") {
      const separator =
        /\s$/.test(previous.text) || /^\s/.test(text) ? "" : "\n\n";
      previous.text += separator + text;
      previous.key += `-${part.part_seq}`;
    } else {
      blocks.push({ kind: "markdown", key: `text-${part.part_seq}`, text });
    }
  }
  return blocks;
}
