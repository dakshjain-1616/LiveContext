import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, 
  User, 
  Settings, 
  Wrench, 
  Trash2, 
  Merge, 
  Scissors,
  Filter,
  AlertCircle
} from 'lucide-react';
import { ContextMessage, Eviction, EvictionStrategy, Role } from '../types';

interface ContextStreamProps {
  messages: ContextMessage[];
  evictions: Eviction[];
}

export function ContextStream({ messages, evictions }: ContextStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const getRoleIcon = (role: Role) => {
    switch (role) {
      case Role.SYSTEM:
        return <Settings className="w-4 h-4 text-violet-400" />;
      case Role.USER:
        return <User className="w-4 h-4 text-emerald-400" />;
      case Role.ASSISTANT:
        return <Bot className="w-4 h-4 text-sky-400" />;
      case Role.TOOL:
        return <Wrench className="w-4 h-4 text-amber-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-slate-400" />;
    }
  };

  const getRoleColor = (role: Role) => {
    switch (role) {
      case Role.SYSTEM:
        return 'border-violet-500/30 bg-violet-500/10';
      case Role.USER:
        return 'border-emerald-500/30 bg-emerald-500/10';
      case Role.ASSISTANT:
        return 'border-sky-500/30 bg-sky-500/10';
      case Role.TOOL:
        return 'border-amber-500/30 bg-amber-500/10';
      default:
        return 'border-slate-500/30 bg-slate-500/10';
    }
  };

  const getStrategyIcon = (strategy: EvictionStrategy) => {
    switch (strategy) {
      case EvictionStrategy.SLIDING_WINDOW:
        return <Scissors className="w-3.5 h-3.5" />;
      case EvictionStrategy.SIMILARITY_MERGE:
        return <Merge className="w-3.5 h-3.5" />;
      case EvictionStrategy.TOKEN_TRUNCATION:
        return <Scissors className="w-3.5 h-3.5" />;
      case EvictionStrategy.IMPORTANCE_FILTER:
        return <Filter className="w-3.5 h-3.5" />;
      default:
        return <Trash2 className="w-3.5 h-3.5" />;
    }
  };

  const getStrategyColor = (strategy: EvictionStrategy) => {
    switch (strategy) {
      case EvictionStrategy.SLIDING_WINDOW:
        return 'bg-slate-500/20 text-slate-400';
      case EvictionStrategy.SIMILARITY_MERGE:
        return 'bg-violet-500/20 text-violet-400';
      case EvictionStrategy.TOKEN_TRUNCATION:
        return 'bg-amber-500/20 text-amber-400';
      case EvictionStrategy.IMPORTANCE_FILTER:
        return 'bg-sky-500/20 text-sky-400';
      default:
        return 'bg-slate-500/20 text-slate-400';
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const truncateContent = (content: string, maxLength: number = 200) => {
    if (content.length <= maxLength) return content;
    return content.slice(0, maxLength) + '...';
  };

  // Combine messages and evictions for display
  const items = [
    ...messages.map(m => ({ type: 'message' as const, data: m, timestamp: m.timestamp })),
    ...evictions.map(e => ({ type: 'eviction' as const, data: e, timestamp: e.timestamp }))
  ].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden"
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Bot className="w-4 h-4 text-sky-500" />
            Context Stream
          </h2>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              {messages.filter(m => m.role === Role.USER).length} user
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-sky-500" />
              {messages.filter(m => m.role === Role.ASSISTANT).length} assistant
            </span>
            {evictions.length > 0 && (
              <span className="flex items-center gap-1 text-amber-400">
                <Trash2 className="w-3 h-3" />
                {evictions.length} evicted
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Stream */}
      <div 
        ref={scrollRef}
        className="max-h-[500px] overflow-y-auto p-4 space-y-3"
      >
        <AnimatePresence>
          {items.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12 text-slate-500"
            >
              <Bot className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="text-sm">No messages in context yet</p>
              <p className="text-xs mt-1">Messages will appear here as they are sent</p>
            </motion.div>
          ) : (
            items.map((item, index) => (
              <motion.div
                key={`${item.type}-${item.data.id}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
              >
                {item.type === 'message' ? (
                  // Message item
                  <div className={`group p-3 rounded-lg border ${getRoleColor((item.data as ContextMessage).role)} transition-all hover:brightness-110`}>
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">
                        {getRoleIcon((item.data as ContextMessage).role)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-medium text-slate-300 capitalize">
                            {(item.data as ContextMessage).role}
                          </span>
                          <span className="text-xs text-slate-500">
                            {formatTime(item.timestamp)}
                          </span>
                          <span className="text-xs text-slate-500">
                            {(item.data as ContextMessage).token_count} tokens
                          </span>
                          {(item.data as ContextMessage).importance_score < 1 && (
                            <span className="text-xs text-amber-400">
                              importance: {(item.data as ContextMessage).importance_score.toFixed(2)}
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-slate-200 leading-relaxed">
                          {truncateContent((item.data as ContextMessage).content)}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  // Eviction item
                  <div className="flex items-center gap-3 p-3 rounded-lg border border-amber-500/20 bg-amber-500/5">
                    <div className={`p-1.5 rounded ${getStrategyColor((item.data as Eviction).strategy)}`}>
                      {getStrategyIcon((item.data as Eviction).strategy)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-amber-400">
                          Message Evicted
                        </span>
                        <span className="text-xs text-slate-500">
                          {formatTime(item.timestamp)}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {(item.data as Eviction).reason}
                      </p>
                      {(item.data as Eviction).similarity_score && (
                        <p className="text-xs text-slate-500 mt-0.5">
                          Similarity: {((item.data as Eviction).similarity_score! * 100).toFixed(1)}%
                        </p>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-medium text-emerald-400">
                        -{(item.data as Eviction).token_savings} tokens
                      </span>
                    </div>
                  </div>
                )}
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-slate-800 bg-slate-900/30">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Total: {messages.length} messages</span>
          <span>Evicted: {evictions.length}</span>
        </div>
      </div>
    </motion.div>
  );
}
