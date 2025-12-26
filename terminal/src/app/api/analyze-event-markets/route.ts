import { NextRequest, NextResponse } from "next/server";
import type { AnalyzeMarketRequest, AnalyzeMarketResponse } from "@/types/api";

/**
 * Server-side API route to proxy requests to the Supabase Edge Function.
 * This keeps the Supabase URL and keys secure on the server.
 */
export async function POST(request: NextRequest) {
  try {
    // Read environment variables server-side
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.json(
        {
          success: false,
          error: "Server configuration error: Missing Supabase credentials",
        },
        { status: 500 }
      );
    }

    // Parse request body
    const body: AnalyzeMarketRequest = await request.json();

    // Validate required fields
    if (!body.url) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: url",
        },
        { status: 400 }
      );
    }

    if (!body.question) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: question",
        },
        { status: 400 }
      );
    }

    // Auto-detect pmType based on URL if not provided
    const pmType = body.pmType || (body.url.includes("kalshi.com") ? "Kalshi" : "Polymarket");

    // Call the Supabase Edge Function
    const edgeFunctionUrl = process.env.SUPABASE_EDGE_FUNCTION_ANALYZE_EVENT_MARKETS 
      || `${supabaseUrl}/functions/v1/analyze-event-markets`;
    
    const response = await fetch(edgeFunctionUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${supabaseAnonKey}`,
        apikey: supabaseAnonKey,
      },
      body: JSON.stringify({
        url: body.url,
        question: body.question,
        pmType,
        model: body.model,
        dataProvider: body.dataProvider || 'dome',
      }),
    });

    const data: AnalyzeMarketResponse = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Error in analyze-event-markets API route:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "An unexpected error occurred",
      },
      { status: 500 }
    );
  }
}

