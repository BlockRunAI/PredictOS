"use client";

import { useState } from "react";
import Image from "next/image";
import TerminalInput, { type AIModel } from "./TerminalInput";
import AnalysisOutput from "./AnalysisOutput";
import type { MarketAnalysis, AnalyzeMarketResponse } from "@/types/api";

interface AnalysisResult {
  id: string;
  analysis: MarketAnalysis;
  timestamp: Date;
  marketUrl?: string;
}

/**
 * Validate that the URL is a supported prediction market URL (Kalshi or Polymarket)
 */
function isPredictionMarketUrl(url: string): boolean {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname.includes("kalshi") || urlObj.hostname.includes("polymarket");
  } catch {
    return false;
  }
}

const Terminal = () => {
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shouldClearInput, setShouldClearInput] = useState(false);

  const handleSubmit = async (url: string, model: AIModel) => {
    setIsLoading(true);
    setError(null);
    setShouldClearInput(false);
    
    try {
      // Validate the URL
      if (!isPredictionMarketUrl(url)) {
        throw new Error("Invalid URL. Please paste a valid Kalshi or Polymarket URL.");
      }

      // Call our server-side API
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          question: "What is the best trading opportunity in this market? Analyze the probability and provide a recommendation.",
          model,
        }),
      });

      const data: AnalyzeMarketResponse = await response.json();

      if (!data.success || !data.data) {
        throw new Error(data.error || "Failed to analyze market");
      }

      const result: AnalysisResult = {
        id: data.metadata.requestId,
        analysis: data.data,
        timestamp: new Date(data.metadata.timestamp),
        marketUrl: data["pm-market-url"],
      };

      setAnalyses(prev => [result, ...prev]);
      setShouldClearInput(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-80px)] px-2 py-4 md:px-4 md:py-6">
      <div className="max-w-4xl mx-auto">
        <div className="space-y-6">
          {/* Header - Always visible */}
          <div className="text-center py-8 fade-in">
            {/* AI Market Analysis */}
            <div className="relative mb-8">
              <h2 className="font-display text-xl md:text-2xl font-bold text-primary text-glow mb-3">
                AI Market Analysis
              </h2>
              <p className="text-muted-foreground max-w-lg mx-auto">
                Paste a Kalshi or Polymarket URL to get instant AI-powered analysis 
                with probability estimates and alpha opportunities.
              </p>
            </div>

            {/* Powered by Dome */}
            <div>
              <a 
                href="https://domeapi.io/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/50 border border-border/50 text-xs text-muted-foreground hover:text-primary hover:border-primary/50 transition-colors"
              >
                <Image 
                  src="/dome-icon-light.svg" 
                  alt="Dome" 
                  width={16} 
                  height={16} 
                  className="w-4 h-4"
                />
                <span>Powered by Dome</span>
              </a>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="border border-destructive/50 rounded-lg bg-destructive/10 p-4 fade-in">
              <p className="text-destructive text-sm font-mono">{`> Error: ${error}`}</p>
            </div>
          )}

          {/* Input */}
          <TerminalInput onSubmit={handleSubmit} isLoading={isLoading} shouldClear={shouldClearInput} />

          {/* Analysis Results */}
          <div className="space-y-4">
            {analyses.map((result) => (
              <AnalysisOutput
                key={result.id}
                analysis={result.analysis}
                timestamp={result.timestamp}
                marketUrl={result.marketUrl}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Terminal;

