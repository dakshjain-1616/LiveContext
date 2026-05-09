export enum Role {
  SYSTEM = 'system',
  USER = 'user',
  ASSISTANT = 'assistant',
  TOOL = 'tool'
}

export enum EvictionStrategy {
  TOKEN_TRUNCATION = 'token_truncation',
  SIMILARITY_MERGE = 'similarity_merge',
  IMPORTANCE_FILTER = 'importance_filter',
  SLIDING_WINDOW = 'sliding_window',
  MANUAL = 'manual'
}

export interface ContextMessage {
  id: string;
  role: Role;
  content: string;
  timestamp: string;
  token_count: number;
  embedding?: number[];
  importance_score: number;
  metadata?: Record<string, any>;
}

export interface Eviction {
  id: string;
  message_id: string;
  strategy: EvictionStrategy;
  timestamp: string;
  reason: string;
  token_savings: number;
  similarity_score?: number;
  merged_into?: string;
}

export interface ContextSnapshot {
  id: string;
  session_id: string;
  timestamp: string;
  messages: ContextMessage[];
  evictions: Eviction[];
  total_tokens: number;
  max_tokens: number;
  utilization_percent: number;
  model_name: string;
  provider: string;
}

export interface SessionInfo {
  id: string;
  created_at: string;
  updated_at: string;
  model_name: string;
  provider: string;
  max_tokens: number;
  message_count: number;
  total_evictions: number;
  is_active: boolean;
}

export interface StreamingEvent {
  event_type: 'snapshot' | 'eviction' | 'message' | 'error';
  payload: ContextSnapshot | Eviction | ContextMessage | any;
  timestamp: string;
  session_id: string;
}
