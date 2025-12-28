/**
 * Generates prompts for the bookmaker agent to aggregate multiple agent analyses into a consolidated view.
 */

import type { MarketAnalysis } from "../../../event-analysis-agent/types.ts";

export interface AgentAnalysis {
  agentId: string;
  model: string;
  analysis: MarketAnalysis;
}

export function bookmakerAnalysisPrompt(
  analyses: AgentAnalysis[],
  eventIdentifier: string,
  pmType: string
): {
  systemPrompt: string;
  userPrompt: string;
} {
  const systemPrompt = `You are a senior financial analyst who specializes in synthesizing multiple expert opinions on prediction markets.
Your task is to combine and consolidate analyses from multiple AI agents into a single, authoritative assessment.
You weigh each agent's analysis based on their confidence levels and the consistency of their reasoning.
When agents disagree, you provide balanced perspective on both sides before making a final recommendation.
Your output is ALWAYS in JSON format and you are VERY STRICT about it. You must return valid JSON that matches the exact schema specified.`;

  const analysesText = analyses.map((a, i) => `
### Agent ${i + 1}: ${a.model}
- **Ticker**: ${a.analysis.ticker}
- **Title**: ${a.analysis.title}
- **Market Probability**: ${a.analysis.marketProbability}%
- **Estimated Actual Probability**: ${a.analysis.estimatedActualProbability}%
- **Alpha Opportunity**: ${a.analysis.alphaOpportunity > 0 ? "+" : ""}${a.analysis.alphaOpportunity}%
- **Has Alpha**: ${a.analysis.hasAlpha}
- **Predicted Winner**: ${a.analysis.predictedWinner}
- **Winner Confidence**: ${a.analysis.winnerConfidence}%
- **Recommended Action**: ${a.analysis.recommendedAction}
- **Confidence**: ${a.analysis.confidence}%
- **Reasoning**: ${a.analysis.reasoning}
- **Key Factors**: ${a.analysis.keyFactors.join("; ")}
- **Risks**: ${a.analysis.risks.join("; ")}
- **Analysis Summary**: ${a.analysis.analysisSummary}
`).join("\n");

  const userPrompt = `# Task: Aggregate Multiple Agent Analyses

You are consolidating analyses from ${analyses.length} AI agents for the ${pmType} event: ${eventIdentifier}

## Individual Agent Analyses
${analysesText}

## Your Task

Review all agent analyses and create a consolidated assessment that:
1. **Identifies consensus**: Where do agents agree?
2. **Highlights disagreements**: Where do agents disagree, and what are the arguments on each side?
3. **Weighs confidence**: Give more weight to high-confidence analyses with strong reasoning
4. **Synthesizes findings**: Create a final recommendation that accounts for all perspectives

## Output Format

Return your consolidated analysis in JSON format:

{
  "event_ticker": "string - event identifier",
  "ticker": "string - the market ticker most agents recommend or the best opportunity",
  "title": "string - market title",
  "marketProbability": number - consensus market probability (0-100),
  "estimatedActualProbability": number - consolidated estimated probability (0-100),
  "alphaOpportunity": number - consolidated alpha assessment,
  "hasAlpha": boolean - whether consolidated view shows meaningful alpha,
  "predictedWinner": "string - either 'YES' or 'NO'",
  "winnerConfidence": number - consolidated confidence (0-100),
  "recommendedAction": "string - either 'BUY YES', 'BUY NO', or 'NO TRADE'",
  "reasoning": "string - synthesized reasoning that accounts for all agent perspectives, noting agreements and disagreements",
  "confidence": number - overall consolidated confidence (0-100),
  "keyFactors": ["string"] - combined and deduplicated key factors from all agents,
  "risks": ["string"] - combined and deduplicated risks from all agents,
  "questionAnswer": "string - consolidated answer based on all agent analyses",
  "analysisSummary": "string - brief consolidated summary under 270 characters",
  "agentConsensus": {
    "agreementLevel": "string - 'high' (>80% agree), 'medium' (50-80%), or 'low' (<50%)",
    "majorityRecommendation": "string - what most agents recommended",
    "dissenting": ["string"] - any dissenting opinions summarized
  }
}

## Important Notes

- If agents strongly disagree (low consensus), be more conservative with confidence
- Highlight when there's unanimous agreement vs split opinions
- The agentConsensus field helps users understand how aligned the agents were
- Your reasoning should explicitly mention which agents agreed/disagreed and why
- Be balanced - don't ignore minority opinions if they have valid points

Now consolidate these analyses and provide your synthesized assessment.`;

  return {
    systemPrompt,
    userPrompt,
  };
}

