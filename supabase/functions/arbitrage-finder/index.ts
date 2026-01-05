/**
 * Supabase Edge Function: arbitrage-finder
 * 
 * Finds arbitrage opportunities across Polymarket and Kalshi markets.
 * 
 * Flow:
 * 1. Parse input URL to determine source platform (Polymarket or Kalshi)
 * 2. Fetch event data: title + markets (title + yes price only)
 * 3. Use AI agent to generate 1-2 word search query from event title
 * 4. Search the OTHER platform using the generated query
 * 5. If no results, return early
 * 6. Pass source markets + search results to arbitrage analysis agent
 * 7. Return results to frontend
 */

import { arbitrageAnalysisPrompt } from "../_shared/ai/prompts/arbitrageAnalysis.ts";
import { searchQueryGeneratorPrompt } from "../_shared/ai/prompts/searchQueryGenerator.ts";
import { callGrokResponses } from "../_shared/ai/callGrok.ts";
import { callOpenAIResponses } from "../_shared/ai/callOpenAI.ts";
import type { GrokMessage, GrokOutputText, OpenAIMessage, OpenAIOutputText } from "../_shared/ai/types.ts";
import { request as dflowRequest } from "../_shared/dflow/client.ts";
import type {
  ArbitrageRequest,
  ArbitrageResponse,
  ArbitrageMarketData,
  ArbitrageAnalysis,
  ArbitrageMarketSource,
} from "./types.ts";

// API endpoints
const GAMMA_API_URL = "https://gamma-api.polymarket.com";
const DFLOW_API_BASE = "https://a.prediction-markets-api.dflow.net/api/v1";

// OpenAI model identifiers
const OPENAI_MODELS = ["gpt-5.2", "gpt-5.1", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini"];

function isOpenAIModel(model: string): boolean {
  return OPENAI_MODELS.includes(model) || model.startsWith("gpt-");
}

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

/** Simplified market data for arbitrage comparison */
interface SimplifiedMarket {
  title: string;
  yesPrice: number;
}

/** Source event data */
interface SourceEventData {
  eventTitle: string;
  markets: SimplifiedMarket[];
  source: ArbitrageMarketSource;
}

// =============================================================================
// URL Parsing
// =============================================================================

/**
 * Detect which platform a URL is from
 */
function detectPlatform(url: string): ArbitrageMarketSource | null {
  const lowerUrl = url.toLowerCase();
  if (lowerUrl.includes('polymarket.com')) return 'polymarket';
  if (lowerUrl.includes('kalshi.com')) return 'kalshi';
  return null;
}

/**
 * Extract event slug from Polymarket URL
 * The event name can be derived by replacing '-' with spaces
 */
function extractPolymarketSlug(url: string): string | null {
  try {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(p => p);
    
    // Format: /event/[event-slug] or /event/[event-slug]/[market-slug]
    if (pathParts[0] === 'event' && pathParts.length >= 2) {
      return pathParts[1]; // Return the event slug
    }
    
    return null;
  } catch {
    return null;
  }
}

/**
 * Extract Kalshi event ticker from URL
 * Format: /markets/[base-ticker]/[slug]/[full-ticker]
 * Returns the full ticker (last segment) uppercased
 */
function extractKalshiTicker(url: string): string | null {
  try {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(p => p);
    
    // Format: /markets/[base-ticker]/[slug]/[full-ticker] or /events/[event-ticker]
    if ((pathParts[0] === 'markets' || pathParts[0] === 'events') && pathParts.length >= 2) {
      // For markets URLs with full ticker at the end (4 segments), use the last segment
      // Otherwise fall back to the second segment
      const ticker = pathParts.length >= 4 ? pathParts[pathParts.length - 1] : pathParts[1];
      return ticker.toUpperCase();
    }
    
    return null;
  } catch {
    return null;
  }
}

// =============================================================================
// Data Fetching
// =============================================================================

/**
 * Fetch Polymarket event data by slug
 * Returns event title + markets (title + yes price only)
 */
async function fetchPolymarketEvent(slug: string): Promise<SourceEventData | null> {
  try {
    // Fetch event by slug
    const eventUrl = `${GAMMA_API_URL}/events/slug/${slug}`;
    const response = await fetch(eventUrl);
    
    if (!response.ok) {
      console.error("Polymarket event fetch error:", response.status);
      return null;
    }
    
    const event = await response.json();
    
    // Extract markets with title + yes price
    const markets: SimplifiedMarket[] = [];
    if (event.markets && Array.isArray(event.markets)) {
      for (const market of event.markets) {
        const title = market.question || market.title || '';
        let yesPrice = 50; // Default
        
        if (market.outcomePrices) {
          try {
            const prices = JSON.parse(market.outcomePrices);
            yesPrice = parseFloat(prices[0]) * 100; // Convert decimal to percentage
          } catch {
            // Keep default
          }
        }
        
        if (title) {
          markets.push({ title, yesPrice });
        }
      }
    }
    
    return {
      eventTitle: event.title || slug.replace(/-/g, ' '),
      markets,
      source: 'polymarket',
    };
  } catch (error) {
    console.error("Error fetching Polymarket event:", error);
    return null;
  }
}

/**
 * Fetch Kalshi event data by ticker via DFlow
 * Returns event title + markets (title + yes price only)
 */
async function fetchKalshiEvent(ticker: string): Promise<SourceEventData | null> {
  try {
    const response = await dflowRequest<{
      event_ticker: string;
      title?: string;
      markets: Array<{
        ticker: string;
        title: string;
        last_price: number;
        yes_bid: number;
        yes_ask: number;
      }>;
    }>(`/event/${ticker}`, {
      params: { withNestedMarkets: true },
    });
    
    // Extract markets with title + yes price
    const markets: SimplifiedMarket[] = [];
    if (response.markets && Array.isArray(response.markets)) {
      for (const market of response.markets) {
        const title = market.title || '';
        // Use last_price or mid of yes_bid/yes_ask
        const yesPrice = market.last_price || ((market.yes_bid + market.yes_ask) / 2);
        
        if (title) {
          markets.push({ title, yesPrice });
        }
      }
    }
    
    return {
      eventTitle: response.title || ticker,
      markets,
      source: 'kalshi',
    };
  } catch (error) {
    console.error("Error fetching Kalshi event:", error);
    return null;
  }
}

// =============================================================================
// Search Functions
// =============================================================================

/**
 * Search Polymarket for events matching query
 * Returns simplified market data (title + yes price)
 */
async function searchPolymarket(query: string): Promise<SimplifiedMarket[]> {
  try {
    const url = `${GAMMA_API_URL}/events?status=open&q=${encodeURIComponent(query)}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      console.error("Polymarket search error:", response.status);
      return [];
    }
    
    const events = await response.json();
    const markets: SimplifiedMarket[] = [];
    
    // Extract all markets from all events
    if (Array.isArray(events)) {
      for (const event of events) {
        if (event.markets && Array.isArray(event.markets)) {
          for (const market of event.markets) {
            const title = market.question || market.title || '';
            let yesPrice = 50;
            
            if (market.outcomePrices) {
              try {
                const prices = JSON.parse(market.outcomePrices);
                yesPrice = parseFloat(prices[0]) * 100;
              } catch {
                // Keep default
              }
            }
            
            if (title) {
              markets.push({ title, yesPrice });
            }
          }
        }
      }
    }
    
    return markets;
  } catch (error) {
    console.error("Error searching Polymarket:", error);
    return [];
  }
}

/**
 * Search Kalshi for events matching query via DFlow
 * Returns simplified market data (title + yes price)
 */
async function searchKalshi(query: string): Promise<SimplifiedMarket[]> {
  try {
    const response = await dflowRequest<{
      cursor?: number;
      events: Array<{
        ticker: string;
        title: string;
        markets?: Array<{
          ticker: string;
          title: string;
          yesAsk?: string;
          yesBid?: string;
        }>;
      }>;
    }>('/search', {
      params: {
        q: query,
        event_status: 'open',
        withNestedMarkets: true,
      },
    });
    
    const markets: SimplifiedMarket[] = [];
    
    if (response.events && Array.isArray(response.events)) {
      for (const event of response.events) {
        if (event.markets && Array.isArray(event.markets)) {
          for (const market of event.markets) {
            const title = market.title || '';
            // Calculate yes price from yesAsk/yesBid
            let yesPrice = 50;
            if (market.yesAsk && market.yesBid) {
              const ask = parseFloat(market.yesAsk);
              const bid = parseFloat(market.yesBid);
              yesPrice = (ask + bid) / 2;
            } else if (market.yesAsk) {
              yesPrice = parseFloat(market.yesAsk);
            } else if (market.yesBid) {
              yesPrice = parseFloat(market.yesBid);
            }
            
            if (title) {
              markets.push({ title, yesPrice });
            }
          }
        }
      }
    }
    
    return markets;
  } catch (error) {
    console.error("Error searching Kalshi:", error);
    return [];
  }
}

// =============================================================================
// AI Functions
// =============================================================================

/**
 * Generate a 1-2 word search query using AI
 */
async function generateSearchQuery(
  title: string,
  sourcePlatform: ArbitrageMarketSource,
  targetPlatform: ArbitrageMarketSource,
  model: string
): Promise<string> {
  const { systemPrompt, userPrompt } = searchQueryGeneratorPrompt({
    title,
    sourcePlatform,
    targetPlatform,
  });
  
  const useOpenAI = isOpenAIModel(model);
  let text: string;
  
  if (useOpenAI) {
    const response = await callOpenAIResponses(
      userPrompt,
      systemPrompt,
      "text",
      model,
      1 // Low max_tokens since we only want 1-2 words
    );
    
    const content: OpenAIOutputText[] = [];
    for (const item of response.output) {
      if (item.type === "message") {
        const messageItem = item as OpenAIMessage;
        content.push(...messageItem.content);
      }
    }
    
    text = content
      .map((item) => item.text)
      .filter((t) => t !== undefined)
      .join("")
      .trim();
  } else {
    const response = await callGrokResponses(
      userPrompt,
      systemPrompt,
      "text",
      model,
      1
    );
    
    const content: GrokOutputText[] = [];
    for (const item of response.output) {
      if (item.type === "message") {
        const messageItem = item as GrokMessage;
        content.push(...messageItem.content);
      }
    }
    
    text = content
      .map((item) => item.text)
      .filter((t) => t !== undefined)
      .join("")
      .trim();
  }
  
  // Clean up - remove quotes, limit to first 2 words
  const cleaned = text.replace(/['"]/g, '').trim();
  const words = cleaned.split(/\s+/).slice(0, 2);
  return words.join(' ');
}

/**
 * Call the arbitrage analysis AI agent
 */
async function analyzeArbitrage(
  sourceEvent: SourceEventData,
  searchResults: SimplifiedMarket[],
  searchPlatform: ArbitrageMarketSource,
  model: string
): Promise<{
  analysis: ArbitrageAnalysis;
  modelUsed: string;
  tokensUsed?: number;
}> {
  // Convert source event to ArbitrageMarketData format for the prompt
  const sourceMarket: ArbitrageMarketData = {
    source: sourceEvent.source,
    name: sourceEvent.eventTitle,
    identifier: sourceEvent.eventTitle,
    yesPrice: sourceEvent.markets[0]?.yesPrice || 50,
    noPrice: 100 - (sourceEvent.markets[0]?.yesPrice || 50),
    url: '',
  };
  
  // Include all source markets in raw data
  const rawSourceData = {
    eventTitle: sourceEvent.eventTitle,
    markets: sourceEvent.markets,
  };
  
  sourceMarket.rawData = rawSourceData;
  
  const { systemPrompt, userPrompt } = arbitrageAnalysisPrompt({
    sourceMarket,
    searchResults,
    searchPlatform,
  });
  
  const useOpenAI = isOpenAIModel(model);
  let text: string;
  let modelUsed: string;
  let tokensUsed: number | undefined;
  
  if (useOpenAI) {
    const response = await callOpenAIResponses(
      userPrompt,
      systemPrompt,
      "json_object",
      model,
      3
    );
    
    modelUsed = response.model;
    tokensUsed = response.usage?.total_tokens;
    
    const content: OpenAIOutputText[] = [];
    for (const item of response.output) {
      if (item.type === "message") {
        const messageItem = item as OpenAIMessage;
        content.push(...messageItem.content);
      }
    }
    
    text = content
      .map((item) => item.text)
      .filter((t) => t !== undefined)
      .join("\n");
  } else {
    const response = await callGrokResponses(
      userPrompt,
      systemPrompt,
      "json_object",
      model,
      3
    );
    
    modelUsed = response.model;
    tokensUsed = response.usage?.total_tokens;
    
    const content: GrokOutputText[] = [];
    for (const item of response.output) {
      if (item.type === "message") {
        const messageItem = item as GrokMessage;
        content.push(...messageItem.content);
      }
    }
    
    text = content
      .map((item) => item.text)
      .filter((t) => t !== undefined)
      .join("\n");
  }
  
  const parsed = JSON.parse(text);
  
  // Build ArbitrageAnalysis from AI response
  const analysis: ArbitrageAnalysis = {
    isSameMarket: parsed.isSameMarket,
    sameMarketConfidence: parsed.sameMarketConfidence,
    marketComparisonReasoning: parsed.marketComparisonReasoning,
    polymarketData: sourceEvent.source === 'polymarket' ? sourceMarket : parsed.matchedMarket || undefined,
    kalshiData: sourceEvent.source === 'kalshi' ? sourceMarket : parsed.matchedMarket || undefined,
    arbitrage: parsed.arbitrage,
    summary: parsed.summary,
    risks: parsed.risks,
    recommendation: parsed.recommendation,
  };
  
  return { analysis, modelUsed, tokensUsed };
}

// =============================================================================
// Main Handler
// =============================================================================

Deno.serve(async (req: Request) => {
  const startTime = Date.now();

  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  console.log("Arbitrage finder received request:", req.method);

  try {
    // Validate request method
    if (req.method !== "POST") {
      return new Response(
        JSON.stringify({ success: false, error: "Method not allowed. Use POST." }),
        { status: 405, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Parse request body
    let requestBody: ArbitrageRequest;
    try {
      requestBody = await req.json();
    } catch {
      return new Response(
        JSON.stringify({ success: false, error: "Invalid JSON in request body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const { url, model } = requestBody;

    // Validate required parameters
    if (!url) {
      return new Response(
        JSON.stringify({ success: false, error: "Missing required parameter: 'url'" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (!model) {
      return new Response(
        JSON.stringify({ success: false, error: "Missing required parameter: 'model'" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Step 1: Detect platform and fetch source event data
    const sourcePlatform = detectPlatform(url);
    if (!sourcePlatform) {
      return new Response(
        JSON.stringify({ 
          success: false, 
          error: "Invalid URL. Must be a Polymarket or Kalshi market URL." 
        }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log("Detected source platform:", sourcePlatform);

    let sourceEvent: SourceEventData | null = null;
    const searchPlatform: ArbitrageMarketSource = sourcePlatform === 'polymarket' ? 'kalshi' : 'polymarket';

    if (sourcePlatform === 'polymarket') {
      const slug = extractPolymarketSlug(url);
      if (!slug) {
        return new Response(
          JSON.stringify({ success: false, error: "Could not extract event slug from Polymarket URL" }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      console.log("Extracted Polymarket slug:", slug);
      sourceEvent = await fetchPolymarketEvent(slug);
    } else {
      const ticker = extractKalshiTicker(url);
      if (!ticker) {
        return new Response(
          JSON.stringify({ success: false, error: "Could not extract ticker from Kalshi URL" }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      console.log("Extracted Kalshi ticker:", ticker);
      sourceEvent = await fetchKalshiEvent(ticker);
    }

    if (!sourceEvent || sourceEvent.markets.length === 0) {
      return new Response(
        JSON.stringify({ 
          success: false, 
          error: `Could not fetch event data from ${sourcePlatform}. The event may not exist or have no markets.` 
        }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log("Fetched source event:", sourceEvent.eventTitle, "with", sourceEvent.markets.length, "markets");

    // Step 2: Generate search query using AI agent
    console.log("Generating search query from title:", sourceEvent.eventTitle);
    const searchQuery = await generateSearchQuery(
      sourceEvent.eventTitle,
      sourcePlatform,
      searchPlatform,
      model
    );
    console.log("Generated search query:", searchQuery);

    // Step 3: Search the other platform
    console.log("Searching", searchPlatform, "for:", searchQuery);
    let searchResults: SimplifiedMarket[];
    
    if (searchPlatform === 'polymarket') {
      searchResults = await searchPolymarket(searchQuery);
    } else {
      searchResults = await searchKalshi(searchQuery);
    }

    console.log("Found", searchResults.length, "markets on", searchPlatform);

    // Step 4: Return early if no search results
    if (searchResults.length === 0) {
      const processingTimeMs = Date.now() - startTime;
      console.log("No search results found. Returning early.");
      
      const response: ArbitrageResponse = {
        success: true,
        data: {
          isSameMarket: false,
          sameMarketConfidence: 0,
          marketComparisonReasoning: `No matching markets found on ${searchPlatform} for query "${searchQuery}"`,
          polymarketData: sourcePlatform === 'polymarket' ? {
            source: 'polymarket',
            name: sourceEvent.eventTitle,
            identifier: sourceEvent.eventTitle,
            yesPrice: sourceEvent.markets[0]?.yesPrice || 50,
            noPrice: 100 - (sourceEvent.markets[0]?.yesPrice || 50),
            url: url,
            rawData: { eventTitle: sourceEvent.eventTitle, markets: sourceEvent.markets },
          } : undefined,
          kalshiData: sourcePlatform === 'kalshi' ? {
            source: 'kalshi',
            name: sourceEvent.eventTitle,
            identifier: sourceEvent.eventTitle,
            yesPrice: sourceEvent.markets[0]?.yesPrice || 50,
            noPrice: 100 - (sourceEvent.markets[0]?.yesPrice || 50),
            url: url,
            rawData: { eventTitle: sourceEvent.eventTitle, markets: sourceEvent.markets },
          } : undefined,
          arbitrage: {
            hasArbitrage: false,
          },
          summary: `No matching markets found on ${searchPlatform}. Cannot determine arbitrage opportunity.`,
          risks: ["No matching market found on the other platform"],
          recommendation: "Try a different event or check if the market exists on both platforms.",
        },
        metadata: {
          requestId: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          processingTimeMs,
          model,
          sourceMarket: sourcePlatform,
          searchedMarket: searchPlatform,
        },
      };

      return new Response(JSON.stringify(response), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Step 5: Analyze arbitrage using AI agent
    console.log("Analyzing arbitrage opportunity...");
    const { analysis, modelUsed, tokensUsed } = await analyzeArbitrage(
      sourceEvent,
      searchResults,
      searchPlatform,
      model
    );

    console.log("AI analysis complete, isSameMarket:", analysis.isSameMarket);

    const processingTimeMs = Date.now() - startTime;
    console.log("Request completed in", processingTimeMs, "ms");

    const response: ArbitrageResponse = {
      success: true,
      data: analysis,
      metadata: {
        requestId: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        processingTimeMs,
        model: modelUsed,
        tokensUsed,
        sourceMarket: sourcePlatform,
        searchedMarket: searchPlatform,
      },
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });

  } catch (error) {
    console.error("Unhandled error:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "An unexpected error occurred",
        metadata: {
          requestId: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          processingTimeMs: Date.now() - startTime,
          model: "unknown",
          sourceMarket: "unknown",
          searchedMarket: "unknown",
        },
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
