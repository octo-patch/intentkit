# Release v2.42.0

## Model Catalog

- **Model ids no longer carry a version.** An agent now picks "Gemini Flash", "GPT Luna" or "Grok" as a series, and the catalog decides which release serves it. When a provider ships a new version, agents move over automatically and nothing in their configuration changes. Agents created before this release are recognised by their old ids and show the current entry in the model picker; no migration is needed.
- **OpenRouter follows the latest release by itself.** For the vendors OpenRouter offers a "latest" alias for (Anthropic, OpenAI, Gemini Flash, DeepSeek, xAI, Z.ai, Moonshot), the catalog now points at that alias, so new versions arrive as soon as OpenRouter switches them on.
- **Gemini 3.8 Flash** replaces 3.7 Flash on both Google and OpenRouter, at the same price and with the same thinking controls. Web search grounding uses it too.
- **Muse Spark 1.3 (Contributor)** from Meta is available through OpenRouter: a 1M-context multimodal reasoning model at a very low price. Meta may use requests to improve its models and limits the tier to 100 requests per minute, and the OpenRouter account must have its 18+ confirmation enabled before it can be used.
- **Qwen3.8 Max** moves to Alibaba's September snapshot, which improves coding, agent workflows and image understanding at the same price.
- Prices were re-checked against every pinned endpoint. OpenRouter's discount on GPT-5.6 Terra and Luna has ended and moved to GPT-5.6 Sol, OpenAI has lowered Sol's own price, and GLM 5.3 Flash's launch discount has ended.
- DeepSeek has withdrawn the planned retirement of V4 Pro; it stays available at the same price.

## Security & Maintenance

- PDF rendering now uses the latest rendering engine, which fixes a security advisory and makes blocked internal-network fetches visible in the logs.
- Updated the Anthropic, OpenAI, web3 and other Python dependencies, the frontend framework (closing two critical advisories) and all Go modules.
