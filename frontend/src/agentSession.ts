import type { AgentId, Chat } from "./types";

export function reusableChatId(
  chat: Chat | null,
  selectedAgentId: AgentId,
): string | undefined {
  if (!chat?.chat_id) return undefined;
  return chat.metadata?.agent_id === selectedAgentId ? chat.chat_id : undefined;
}
