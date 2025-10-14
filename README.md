# ghostcode

Terminal-based LLM powered coding assistant based on the popular ghostbox LLM library. Uses a hybrid approach to combine the intelligence of cloud provided LLMs with the efficiency of local models.

Everyone's rolling their own codex-cli right now and this is mine. What's different?

## Features
 - Hybrid approach. Ghostcode uses ghostcoder and ghostworker as LLMs.
     - ghostcoder: High level planning, code generation, code review, brainstorming and discussion. Your senior dev.
     - ghostworker: File edits, shell commands, running tests, Call on to ghostcoder if things go wrong. Junior dev.
 - Use local or cloud LLMs. For either ghostcoder or ghostworker.
 - Supports OpenAI's Chat-GPT, Google's Gemini, counhtless local models with llama.cpp, and all others that support the OpenAI API.
 - Minimalist ethos. ghostcoder seeks to empower developers, not replace them.
 - CLI based with screen-reader aware design.
 - Tight emacs integration (WIP)

## Usage Tips

 - Set the ghostcoder to a powerful cloud provider. I like gemini for the large context.
 - Use llama.cpp's llama-server. The qwen3 family of models  is currently a good place to start.
  - Set ghostworker to target the llama-server endpoint. The ghostcoder prompts are designed to be clutter free and will honor your token count. The ghostworker is less parsemonious and therefore perfect for token-expensive grunt work that is done locally. Hear your RTX 3090 go brrr as it repeatedly tries to do levenshtein distance based file edits.
  
This is very much a work-in-progress, so stay tuned.

## Installation

Coming soon!
