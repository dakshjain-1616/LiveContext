import { motion } from 'framer-motion';
import { Plus, Trash2, RefreshCw, Activity, Clock, Database } from 'lucide-react';
import { SessionInfo } from '../types';

interface SessionSelectorProps {
  sessions: SessionInfo[];
  selectedSession: string | null;
  onSelect: (sessionId: string | null) => void;
  loading?: boolean;
}

export function SessionSelector({ 
  sessions, 
  selectedSession, 
  onSelect,
  loading = false 
}: SessionSelectorProps) {
  const formatDuration = (createdAt: string) => {
    const created = new Date(createdAt);
    const now = new Date();
    const diffMs = now.getTime() - created.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) return `${diffDays}d ago`;
    if (diffHours > 0) return `${diffHours}h ago`;
    if (diffMins > 0) return `${diffMins}m ago`;
    return 'Just now';
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-sky-500" />
            Sessions
          </h2>
          <div className="flex items-center gap-1">
            <button 
              className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4 text-slate-400" />
            </button>
            <button 
              className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors"
              title="New Session"
            >
              <Plus className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Session List */}
      <div className="max-h-[calc(100vh-200px)] overflow-y-auto">
        {loading ? (
          <div className="p-8 text-center">
            <RefreshCw className="w-6 h-6 text-slate-500 animate-spin mx-auto" />
            <p className="mt-2 text-sm text-slate-500">Loading sessions...</p>
          </div>
        ) : sessions.length === 0 ? (
          <div className="p-8 text-center">
            <Database className="w-8 h-8 text-slate-600 mx-auto mb-2" />
            <p className="text-sm text-slate-500">No sessions yet</p>
            <p className="text-xs text-slate-600 mt-1">
              Start monitoring to create a session
            </p>
          </div>
        ) : (
          <div className="divide-y divide-slate-800">
            {sessions.map((session, index) => (
              <motion.button
                key={session.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => onSelect(session.id === selectedSession ? null : session.id)}
                className={`w-full p-4 text-left transition-all hover:bg-slate-800/50 ${
                  session.id === selectedSession 
                    ? 'bg-slate-800/80 border-l-2 border-sky-500' 
                    : 'border-l-2 border-transparent'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${
                        session.is_active ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'
                      }`} />
                      <span className="text-sm font-medium text-slate-200 truncate">
                        {session.model_name}
                      </span>
                    </div>
                    
                    <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                      <span className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">
                        {session.provider}
                      </span>
                      <span>•</span>
                      <span>{session.message_count} msgs</span>
                    </div>

                    <div className="mt-2 flex items-center gap-1 text-xs text-slate-500">
                      <Clock className="w-3 h-3" />
                      {formatDuration(session.created_at)}
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      // Handle delete
                    }}
                    className="p-1.5 opacity-0 group-hover:opacity-100 hover:bg-red-500/10 hover:text-red-400 rounded transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Token usage mini bar */}
                {session.max_tokens > 0 && (
                  <div className="mt-3">
                    <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-sky-500 to-violet-500 rounded-full transition-all"
                        style={{ 
                          width: `${Math.min(100, (session.message_count * 10 / session.max_tokens) * 100)}%` 
                        }}
                      />
                    </div>
                    <div className="mt-1 flex justify-between text-xs text-slate-600">
                      <span>{session.message_count} msgs</span>
                      <span>{session.total_evictions} evictions</span>
                    </div>
                  </div>
                )}
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
