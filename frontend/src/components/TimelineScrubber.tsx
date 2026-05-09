import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, SkipBack, SkipForward, Clock, Calendar } from 'lucide-react';
import { ContextSnapshot } from '../types';

interface TimelineScrubberProps {
  snapshots: ContextSnapshot[];
  currentSnapshot: ContextSnapshot | null;
  onSelect: (snapshot: ContextSnapshot) => void;
}

export function TimelineScrubber({ 
  snapshots, 
  currentSnapshot, 
  onSelect 
}: TimelineScrubberProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sort snapshots by timestamp
  const sortedSnapshots = [...snapshots].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  useEffect(() => {
    if (currentSnapshot) {
      const index = sortedSnapshots.findIndex(s => s.id === currentSnapshot.id);
      if (index !== -1) {
        setCurrentIndex(index);
      }
    }
  }, [currentSnapshot, sortedSnapshots]);

  useEffect(() => {
    if (isPlaying) {
      playRef.current = setInterval(() => {
        setCurrentIndex(prev => {
          const next = (prev + 1) % sortedSnapshots.length;
          onSelect(sortedSnapshots[next]);
          return next;
        });
      }, 1000);
    } else {
      if (playRef.current) {
        clearInterval(playRef.current);
      }
    }

    return () => {
      if (playRef.current) {
        clearInterval(playRef.current);
      }
    };
  }, [isPlaying, sortedSnapshots, onSelect]);

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const formatDuration = (start: string, end: string) => {
    const diff = new Date(end).getTime() - new Date(start).getTime();
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const index = parseInt(e.target.value);
    setCurrentIndex(index);
    onSelect(sortedSnapshots[index]);
  };

  const goToStart = () => {
    setCurrentIndex(0);
    onSelect(sortedSnapshots[0]);
  };

  const goToEnd = () => {
    const last = sortedSnapshots.length - 1;
    setCurrentIndex(last);
    onSelect(sortedSnapshots[last]);
  };

  if (sortedSnapshots.length === 0) return null;

  const current = sortedSnapshots[currentIndex];
  const first = sortedSnapshots[0];
  const last = sortedSnapshots[sortedSnapshots.length - 1];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-900/50 border border-slate-800 rounded-xl p-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-sky-500" />
          <span className="text-sm font-medium text-slate-200">Timeline</span>
          <span className="text-xs text-slate-500">
            ({sortedSnapshots.length} snapshots)
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Calendar className="w-3.5 h-3.5" />
          {formatDuration(first.timestamp, last.timestamp)}
        </div>
      </div>

      {/* Timeline visualization */}
      <div ref={containerRef} className="relative h-16 mb-4">
        {/* Timeline bar */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2 bg-slate-800 rounded-full">
          {/* Progress fill */}
          <div 
            className="h-full bg-gradient-to-r from-sky-500 to-violet-500 rounded-full transition-all"
            style={{ width: `${(currentIndex / (sortedSnapshots.length - 1)) * 100}%` }}
          />
        </div>

        {/* Snapshot markers */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex justify-between">
          {sortedSnapshots.map((snapshot, index) => {
            const isActive = index === currentIndex;
            const utilization = snapshot.utilization_percent;
            
            return (
              <motion.button
                key={snapshot.id}
                onClick={() => {
                  setCurrentIndex(index);
                  onSelect(snapshot);
                }}
                className={`relative w-3 h-3 rounded-full transition-all ${
                  isActive 
                    ? 'bg-white scale-125 ring-2 ring-sky-500' 
                    : utilization > 75 
                      ? 'bg-amber-500' 
                      : 'bg-slate-600 hover:bg-slate-500'
                }`}
                whileHover={{ scale: 1.2 }}
                title={`${formatTime(snapshot.timestamp)} - ${utilization.toFixed(1)}%`}
              >
                {isActive && (
                  <motion.div
                    layoutId="active-marker"
                    className="absolute -inset-1 bg-sky-500/30 rounded-full"
                  />
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Scrubber slider */}
      <input
        type="range"
        min={0}
        max={sortedSnapshots.length - 1}
        value={currentIndex}
        onChange={handleScrub}
        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
      />

      {/* Controls */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-800">
        <div className="flex items-center gap-2">
          <button
            onClick={goToStart}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            title="Go to start"
          >
            <SkipBack className="w-4 h-4 text-slate-400" />
          </button>
          
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 bg-sky-500 hover:bg-sky-600 rounded-lg transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 text-white" />
            ) : (
              <Play className="w-4 h-4 text-white" />
            )}
          </button>
          
          <button
            onClick={goToEnd}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            title="Go to end"
          >
            <SkipForward className="w-4 h-4 text-slate-400" />
          </button>
        </div>

        {/* Current time display */}
        <div className="text-right">
          <div className="text-sm font-medium text-slate-200">
            {formatTime(current.timestamp)}
          </div>
          <div className="text-xs text-slate-500">
            {current.total_tokens} tokens • {current.utilization_percent.toFixed(1)}%
          </div>
        </div>
      </div>
    </motion.div>
  );
}
