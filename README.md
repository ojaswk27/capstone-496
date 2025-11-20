Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report

## Title: AI-Powered Aerospace Design Assistant: Automated Design Generation for Any Flying Vehicle

## Overview

My project is an AI assistant that designs flying vehicles automatically. You give it requirements like "I need a drone that flies for 30 minutes with 2kg payload" or "design a small rocket to reach 1km altitude", and it figures out what type of vehicle you need, searches through research papers to find the right formulas and design methods, does all the math calculations, and gives you a complete design with component specifications.

It uses LangGraph to manage the whole process: first it figures out what kind of vehicle you're asking for (drone, plane, helicopter, rocket, satellite, etc.), then searches for relevant papers, extracts the important formulas, calls calculation tools to do the math, checks if the design works, and outputs everything in a nice structured format with citations to the papers it used.

## Reason for picking up this project

This project covers all the main topics we learned in MAT496:

**Prompting**: I use prompts to tell the LLM how to extract equations from research papers, understand what the user wants, and explain why certain design choices were made. Different prompts work for different types of vehicles.

**Structured Output**: All the design specs come out in a clean JSON format with component details, performance numbers, and cost estimates. I'm using Pydantic models to make sure everything is organized properly.

**Semantic Search**: The system searches through aerospace research papers to find relevant information. It filters by vehicle type (drone, plane, rocket, etc.) so it only looks at papers that actually matter for the design problem.

**RAG (Retrieval Augmented Generation)**: After finding the right papers, it pulls out the specific formulas, data, and design methods from them. This information guides which calculations to run, and everything gets cited in the final output.

**Tool calling & MCP**: I built a bunch of calculation tools for different types of vehicles - drones, planes, rockets, helicopters, satellites. The LLM picks which tools to use based on what it's designing. All the actual math happens in these tools using real aerospace formulas.

**Langgraph (State, Nodes, Graph)**: The whole design process is a LangGraph with different nodes for each step: figuring out vehicle type, understanding requirements, searching papers, extracting formulas, picking tools, running calculations, checking if it works, and putting together the final design. The state keeps track of everything as it moves through the graph.

I picked this project because it's creative - instead of just summarizing information, it actually creates new designs by combining knowledge from multiple sources. Plus, it connects to my drone work and lets me apply course concepts to real aerospace problems.

## Plan

I plan to execute these steps to complete my project.

- [TODO] Step 1 involves setting up the project - fork the template repo, install langchain, langgraph, langsmith, and FAISS/ChromaDB, set up API keys

- [TODO] Step 2 involves collecting research papers - download 30-40 papers from arXiv and NASA on drones, planes, helicopters, rockets, and satellites, organize them by vehicle type

- [TODO] Step 3 involves building semantic search - extract text from PDFs, create embeddings, set up vector database, test if search actually finds relevant papers

- [TODO] Step 4 involves making the RAG system - write prompts to pull out formulas and data from papers, add citation tracking so we know where info came from, test on some known equations

- [TODO] Step 5 involves creating all the calculation tools - drone tools (thrust, battery life, propeller size), plane tools (lift, drag, wing area, range), rocket tools (delta-v, staging, burn time), helicopter tools (rotor power), satellite tools (orbit math, power budget), and common tools (weight, balance, stability)

- [TODO] Step 6 involves connecting tools to the LLM - define tool schemas, set up MCP-style registration, make sure the LLM can actually call tools correctly

- [TODO] Step 7 involves designing the state - define what data the graph needs to track (requirements, vehicle type, search results, formulas, calculations, final design), create Pydantic models, add vehicle classification logic

- [TODO] Step 8 involves building all the LangGraph nodes - Vehicle Classifier (what type of vehicle?), Requirement Parser (understand user input), Search Agent (find relevant papers), Extraction Agent (pull out formulas using RAG), Tool Selector (pick the right calculation tools), Calculation Agent (run the math), Validator (does it meet requirements?), Synthesizer (make final design output)

- [TODO] Step 9 involves connecting the graph - add edges between nodes, add routing for different vehicle types, add logic to loop back if validation fails, test the whole flow

- [TODO] Step 10 involves making nice output - define JSON schema for results, create templates for design reports, add tables/charts, include citations

- [TODO] Step 11 involves testing everything - test with surveillance drone (60min flight, 2kg payload), small plane (2 people, 500km range), model rocket (1km altitude), LEO satellite (100kg, 400km orbit), VTOL aircraft, check if answers make sense

- [TODO] Step 12 involves debugging with Langsmith - set up tracing, find slow parts, improve prompts, fix tool calling issues

- [TODO] Step 13 involves writing documentation - README with examples, explain what each node does, add code comments, show example outputs

- [TODO] Step 14 involves final prep - make demo video, finish conclusion, change all TODOs to DONE as I complete them, make sure commits are spread across multiple days

## Conclusion:

I planned to build an AI design assistant that covers all the MAT496 topics (prompting, structured output, semantic search, RAG, tool calling, and LangGraph) and shows creativity by actually generating designs instead of just summarizing information. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.
