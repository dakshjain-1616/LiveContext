import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Terminal, Settings, Info } from 'lucide-react'
import { SessionSelector } from './components/SessionSelector'
import { ContextStream } from './components/ContextStream'
import { TokenGauge } from './components/TokenGauge'
import { TimelineScrubber } from './components/TimelineScrubber'
import { useWebSocket } from './hooks/useWebSocket'
import { useSessions } from './hooks/useSessions'
import { ContextSnapshot } from './types'

function App() {
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [currentSnapshot, setCurrentSnapshot] = useState<ContextSnapshot | null>(null)
  const [snapshots, setSnapshots] = useState<ContextSnapshot[]>([])
  
  const { sessions, loading: sessionsLoading } = useSessions()
  const { lastMessage, connected } = useWebSocket(selectedSession)

  // Load snapshots when session changes
  useEffect(() => {
    if (selectedSession) {
      fetch(`/api/sessions/${selectedSession}/snapshots?limit=100`)
        .then(res => res.json())
        .then(data => {
          setSnapshots(data)
          if (data.length > 0) {
            setCurrentSnapshot(data[data.length - 1])
          }
        })
        .catch(console.error)
    }
  }, [selectedSession])

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.event_type === 'snapshot') {
        const snapshot = lastMessage.payload as ContextSnapshot
        setCurrentSnapshot(snapshot)
        setSnapshots(prev => [...prev, snapshot])
      }
    }
  }, [lastMessage])

  const handleSnapshotSelect = (snapshot: ContextSnapshot) => {
    setCurrentSnapshot(snapshot)
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-sky-500 to-violet-500 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gradient">LiveContext</h1>
                <p className="text-xs text-slate-400">Real-time LLM Context Monitor</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
                <span className="text-sm text-slate-400">
                  {connected ? 'Live' : 'Disconnected'}
                </span>
              </div>
              
              <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                <Terminal className="w-5 h-5 text-slate-400" />
              </button>
              <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                <Settings className="w-5 h-5 text-slate-400" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar - Session Selector */}
          <div className="col-span-3">
            <SessionSelector
              sessions={sessions}
              selectedSession={selectedSession}
              onSelect={setSelectedSession}
              loading={sessionsLoading}
            />
          </div>

          {/* Main Panel */}
          <div className="col-span-9 space-y-6">
            {selectedSession ? (
              <>
                {/* Token Gauge */}
                {currentSnapshot && (
                  <TokenGauge
                    totalTokens={currentSnapshot.total_tokens}
                    maxTokens={currentSnapshot.max_tokens}
                    utilizationPercent={currentSnapshot.utilization_percent}
                    modelName={currentSnapshot.model_name}
                    provider={currentSnapshot.provider}
                  />
                )}

                {/* Timeline Scrubber */}
                {snapshots.length > 0 && (
                  <TimelineScrubber
                    snapshots={snapshots}
                    currentSnapshot={currentSnapshot}
                    onSelect={handleSnapshotSelect}
                  />
                )}

                {/* Context Stream */}
                {currentSnapshot && (
                  <ContextStream
                    messages={currentSnapshot.messages}
                    evictions={currentSnapshot.evictions}
                  />
                )}
              </>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center py-20 text-center"
              >
                <div className="p-4 bg-slate-800/50 rounded-full mb-4">
                  <Info className="w-8 h-8 text-slate-400" />
                </div>
                <h2 className="text-lg font-medium text-slate-300 mb-2">
                  No Session Selected
                </h2>
                <p className="text-slate-400 max-w-md">
                  Select a session from the sidebar to view real-time context window monitoring,
                  or start a new session using the LiveContext SDK.
                </p>
              </motion.div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
