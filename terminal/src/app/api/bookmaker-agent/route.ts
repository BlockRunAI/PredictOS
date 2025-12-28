import { NextRequest, NextResponse } from "next/server";
import type { AnalysisAggregatorRequest, AnalysisAggregatorResponse } from "@/types/agentic";

/**
 * Server-side API route to proxy requests to the bookmaker-agent Edge Function.
 */
export async function POST(request: NextRequest) {
  try {
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

    const body: AnalysisAggregatorRequest = await request.json();

    // Validate required fields
    if (!body.analyses || !Array.isArray(body.analyses) || body.analyses.length < 2) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: analyses (must have at least 2)",
        },
        { status: 400 }
      );
    }

    if (!body.eventIdentifier) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: eventIdentifier",
        },
        { status: 400 }
      );
    }

    if (!body.pmType) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: pmType",
        },
        { status: 400 }
      );
    }

    if (!body.model) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required field: model",
        },
        { status: 400 }
      );
    }

    const edgeFunctionUrl = process.env.SUPABASE_EDGE_FUNCTION_BOOKMAKER_AGENT 
      || `${supabaseUrl}/functions/v1/bookmaker-agent`;
    
    const response = await fetch(edgeFunctionUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${supabaseAnonKey}`,
        apikey: supabaseAnonKey,
      },
      body: JSON.stringify({
        analyses: body.analyses,
        eventIdentifier: body.eventIdentifier,
        pmType: body.pmType,
        model: body.model,
      }),
    });

    const data: AnalysisAggregatorResponse = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Error in bookmaker-agent API route:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "An unexpected error occurred",
      },
      { status: 500 }
    );
  }
}

