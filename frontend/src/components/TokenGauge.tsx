import { motion } from 'framer-motion';
import { Cpu, Layers, AlertTriangle, CheckCircle } from 'lucide-react';

interface TokenGaugeProps {
  totalTokens: number;
  maxTokens: number;
  utilizationPercent: number;
  modelName: string;
  provider: string;
}

export function TokenGauge({ 
  totalTokens, 
  maxTokens, 
  utilizationPercent,
  modelName,
  provider 
}: TokenGaugeProps) {
  const getStatusColor = (percent: number) => {
    if (percent < 50) return 'from-emerald-500 to-emerald-400';
    if (percent < 75) return 'from-amber-500 to-amber-400';
    if (percent < 90) return 'from-orange-500 to-orange-400';
    return 'from-red-500 to-red-400';
  };

  const getStatusIcon = (percent: number) => {
    if (percent < 75) return <CheckCircle className="w-5 h-5 text-emerald-500" />;
    return <AlertTriangle className="w-5 h-5 text-amber-500" />;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toString();
  };

  const circumference = 2 * Math.PI * 120;
  const strokeDashoffset = circumference - (utilizationPercent / 100) * circumference;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-900/50 border border-slate-800 rounded-xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-slate-800 rounded-lg">
            <Cpu className="w-5 h-5 text-sky-500" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-200">Token Usage</h3>
            <p className="text-xs text-slate-500">{modelName} • {provider}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {getStatusIcon(utilizationPercent)}
          <span className={`text-sm font-medium ${
            utilizationPercent > 90 ? 'text-red-400' : 
            utilizationPercent > 75 ? 'text-amber-400' : 'text-emerald-400'
          }`}>
            {utilizationPercent.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="flex items-center justify-center py-4">
        <div className="relative">
          {/* Background circle */}
          <svg className="transform -rotate-90 w-64 h-64">
            <circle
              cx="128"
              cy="128"
              r="120"
              stroke="currentColor"
              strokeWidth="12"
              fill="transparent"
              className="text-slate-800"
            />
            {/* Progress circle */}
            <motion.circle
              cx="128"
              cy="128"
              r="120"
              stroke="currentColor"
              strokeWidth="12"
              fill="transparent"
              strokeLinecap="round"
              className={`bg-gradient-to-r ${getStatusColor(utilizationPercent)}`}
              style={{
                strokeDasharray: circumference,
                strokeDashoffset: strokeDashoffset,
              }}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </svg>
          
          {/* Center content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold text-slate-200">
              {formatNumber(totalTokens)}
            </span>
            <span className="text-sm text-slate-500">/ {formatNumber(maxTokens)}</span>
            <span className="text-xs text-slate-600 mt-1">tokens</span>
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-slate-800">
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-slate-500 mb-1">
            <Layers className="w-3.5 h-3.5" />
            <span className="text-xs">Used</span>
          </div>
          <span className="text-sm font-medium text-slate-300">{formatNumber(totalTokens)}</span>
        </div>
        <div className="text-center border-x border-slate-800">
          <div className="flex items-center justify-center gap-1 text-slate-500 mb-1">
            <span className="text-xs">Remaining</span>
          </div>
          <span className="text-sm font-medium text-slate-300">
            {formatNumber(Math.max(0, maxTokens - totalTokens))}
          </span>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-slate-500 mb-1">
            <span className="text-xs">Efficiency</span>
          </div>
          <span className={`text-sm font-medium ${
            utilizationPercent > 90 ? 'text-red-400' : 
            utilizationPercent > 75 ? 'text-amber-400' : 'text-emerald-400'
          }`}>
            {utilizationPercent.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Warning message */}
      {utilizationPercent > 90 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-xs text-red-400">
              Context window nearly full! Eviction strategies may trigger.
            </span>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
